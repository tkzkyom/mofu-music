import librosa
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import soundfile as sf
import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'  # または 'IPAexGothic', 'Noto Sans CJK JP'
import seaborn as sns

#音源からテンポ（BPM）と拍タイミングの抽出
def extract_bpm_and_beats(audio_path):
    """
    Backwards-compatible wrapper: load audio then delegate to extract_bpm_and_beats_from_loaded.
    This wrapper keeps behavior but avoids UI logic inside the function.
    """
    try:
        y, sr = librosa.load(audio_path, mono=False)
    except Exception as e:
        st.error(f"音声読み込みに失敗しました: {e}")
        raise
    # default to mix
    return extract_bpm_and_beats_from_loaded(y, sr, channel='mix')

# クリック機能は不要のため削除しました


#小節区間の抽出
def get_bar_segments(beat_times, beats_per_bar=4):
    num_bars = len(beat_times) // beats_per_bar
    bar_segments = []
    for i in range(num_bars):
        start = beat_times[i * beats_per_bar]
        if len(beat_times) > (i + 1) * beats_per_bar:
            end = beat_times[(i + 1) * beats_per_bar]
        else:
            end = beat_times[-1]

        bar_segments.append((start, end))
    return bar_segments

#理想の小節
def get_corrected_bar_times(bar_segments, bpm, beats_per_bar=4):
    bar_interval = (60 / bpm) * beats_per_bar
    corrected = [bar_segments[0][0] + i * bar_interval for i in range(len(bar_segments))]
    return corrected


# 自動チャンネル選択（左右のonset強度で判定）
def choose_best_channel(y, sr):
    # y expected shape (2, n)
    try:
        left_env = librosa.onset.onset_strength(y=y[0], sr=sr).mean()
        right_env = librosa.onset.onset_strength(y=y[1], sr=sr).mean()
        return 'left' if left_env > right_env else 'right'
    except Exception:
        return 'mix'


# 音声を既に読み込んだ状態でテンポと拍を抽出する（UIに依存しない）
def extract_bpm_and_beats_from_loaded(y, sr, channel='mix'):
    # y: numpy array, mono or (2, n)
    if isinstance(y, np.ndarray) and y.ndim == 2:
        if channel == 'mix':
            y_mono = np.mean(y, axis=0)
        elif channel == 'left':
            y_mono = y[0]
        else:
            y_mono = y[1]
    else:
        y_mono = y

    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    try:
        tempo = float(tempo)
    except Exception:
        tempo = float(np.array(tempo).item()) if isinstance(tempo, np.ndarray) else float(tempo)
    tempo = round(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) if len(beat_frames) > 0 else np.array([])
    return tempo, y_mono, sr, beat_times

#理想とのずれを抽出
def detect_misaligned_bars(beat_times, bpm, beats_per_bar=4):
    bar_interval = (60 / bpm) * beats_per_bar
    num_bars = len(beat_times) // beats_per_bar
    bar_starts = [beat_times[i * beats_per_bar] for i in range(num_bars)]
    expected_starts = [beat_times[0] + i * bar_interval for i in range(num_bars)]

    deviations = []
    for i, (actual, expected) in enumerate(zip(bar_starts, expected_starts)):
        deviations.append({
            "小節番号": i + 1,
            "理想開始": round(expected, 3),
            "実開始": round(actual, 3),
            "ズレ(ms)": round((actual - expected) * 1000, 1)
        })
    return deviations

#小節からBPMを逆算
def estimate_bar_bpms(bar_segments, beats_per_bar=4):
    bpms = []
    for i, (start, end) in enumerate(bar_segments):
        duration_sec = end - start
        bpm = (beats_per_bar / duration_sec) * 60
        bpms.append({
            "小節番号": i + 1,
            "開始秒": round(start, 3),
            "終了秒": round(end, 3),
            "長さ(s)": round(duration_sec, 3),
            "推定BPM": round(bpm, 2)
        })
    return bpms

def apply_timing_correction(y, sr, bar_segments, corrected_bar_times):
    """
    Apply timing correction per bar. Handles the last bar safely and avoids
    time-stretching very short segments which produce artifacts.
    """
    segments = []
    # estimate ideal bar interval from corrected_bar_times if possible
    if len(corrected_bar_times) > 1:
        bar_interval = float(corrected_bar_times[1] - corrected_bar_times[0])
    else:
        bar_interval = None

    for i in range(len(bar_segments)):
        start_sec = bar_segments[i][0]
        # last bar: end may be the last sample
        if i < len(bar_segments) - 1:
            end_sec = bar_segments[i][1]
        else:
            # if last bar has no explicit end, use the next corrected time if available,
            # else use start + bar_interval if known, else extend to end of audio
            if i < len(corrected_bar_times) - 1:
                end_sec = bar_segments[i][1]
            elif bar_interval is not None:
                end_sec = start_sec + bar_interval
            else:
                # fallback: use end of audio
                end_sec = len(y) / float(sr)

        start_sample = max(0, int(start_sec * sr))
        end_sample = min(int(end_sec * sr), len(y))
        if end_sample <= start_sample:
            continue

        segment = y[start_sample:end_sample]

        actual_duration = float(end_sec - start_sec)
        # determine target duration: prefer corrected_bar_times neighbor, else bar_interval
        if i < len(corrected_bar_times) - 1:
            target_duration = float(corrected_bar_times[i+1] - corrected_bar_times[i])
        elif bar_interval is not None:
            target_duration = bar_interval
        else:
            target_duration = actual_duration

        # rate: old_duration / target_duration (avoid division by zero)
        rate = float(actual_duration) / float(target_duration) if target_duration > 0 else 1.0
        rate = float(np.clip(rate, 0.8, 1.2))

        # avoid stretching very short segments where librosa produces artifacts
        min_duration = 0.08  # seconds
        try:
            if actual_duration < min_duration or target_duration < min_duration:
                # use fix_length to pad/trim to target length
                target_length = max(1, int(round(target_duration * sr)))
                stretched = librosa.util.fix_length(segment.astype(np.float32), size=target_length)
            else:
                stretched = librosa.effects.time_stretch(segment.astype(np.float32), rate)
        except Exception:
            target_length = max(1, int(round(target_duration * sr)))
            stretched = librosa.util.fix_length(segment.astype(np.float32), size=target_length)

        segments.append(stretched)

    if not segments:
        return np.array([])

    # concatenate with crossfade to avoid clicks/artifacts at segment boundaries
    def _make_fade_window(length, kind='hann'):
        if length <= 0:
            return np.array([], dtype=np.float32)
        if kind == 'hann':
            # half Hann window
            w = np.hanning(length * 2)[:length]
            return w.astype(np.float32)
        return np.linspace(0.0, 1.0, length, dtype=np.float32)

    def crossfade_concat(seg_list, sr, cross_ms=20, kind='hann'):
        cross = int(round(sr * cross_ms / 1000.0))
        if cross <= 0:
            return np.concatenate(seg_list)

        fade_in = _make_fade_window(cross, kind)
        fade_out = fade_in[::-1]

        out = seg_list[0].astype(np.float32)
        for seg in seg_list[1:]:
            seg = seg.astype(np.float32)
            if len(out) < cross or len(seg) < cross:
                c = min(len(out), len(seg), cross)
                if c == 0:
                    out = np.concatenate([out, seg])
                    continue
                fi = _make_fade_window(c, kind)
                fo = fi[::-1]
                tail = out[-c:] * fo + seg[:c] * fi
                out = np.concatenate([out[:-c], tail, seg[c:]])
            else:
                tail = out[-cross:] * fade_out + seg[:cross] * fade_in
                out = np.concatenate([out[:-cross], tail, seg[cross:]])
        return out

    return crossfade_concat(segments, sr, cross_ms=20, kind='hann')



def get_corrected_bar_segments(corrected_bar_times):
    segments = []
    for i in range(len(corrected_bar_times) - 1):
        start = corrected_bar_times[i]
        end = corrected_bar_times[i + 1]
        segments.append((start, end))
    return segments

def estimate_corrected_bpms(corrected_segments, beats_per_bar=4):
    bpms = []
    for i, (start, end) in enumerate(corrected_segments):
        duration_sec = end - start
        bpm = (beats_per_bar / duration_sec) * 60
        bpms.append({
            "小節番号": i + 1,
            "開始秒": round(start, 3),
            "終了秒": round(end, 3),
            "長さ(s)": round(duration_sec, 3),
            "補正後BPM": round(bpm, 2)
        })
    return bpms

def plot_bpms(bpms, title="テンポ変化", column="推定BPM"):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [d["小節番号"] for d in bpms]
    y = [d[column] for d in bpms]
    sns.lineplot(x=x, y=y, marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("小節番号")
    ax.set_ylabel("BPM")
    st.pyplot(fig)

st.title("🐈 もふもふミュージック～テンポ抽出～🎵")
uploaded_file = st.file_uploader("音源ファイルをアップロード", type=["mp3", "wav"])

if uploaded_file:
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        # ファイルを一度読み込んで保存
        audio_bytes = uploaded_file.read()
        audio_buffer = BytesIO(audio_bytes)

        # まず音声を読み込む（mono=False で読み込み）
        try:
            y, sr = librosa.load(audio_buffer, mono=False)
        except Exception as e:
            # fallback: write to tempfile and load
            import tempfile, os
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.' + file_ext, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp.flush()
                    tmp_path = tmp.name
                y, sr = librosa.load(tmp_path, mono=False)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # チャンネル選択 UI
        channel = 'mix'
        if isinstance(y, np.ndarray) and y.ndim == 2:
            auto_select = st.checkbox("自動チャンネル選択を有効にする", value=True)
            if auto_select:
                auto_ch = choose_best_channel(y, sr)
                st.info(f"自動選択: {auto_ch}")
                channel = auto_ch
            if st.checkbox("詳細オプションを表示"):
                choice = st.radio("使用チャンネル", ["平均（モノラル）", "左", "右"])
                mapping = {"平均（モノラル）": 'mix', "左": 'left', "右": 'right'}
                channel = mapping[choice]

        # librosaやpydubで使う場合は audio_buffer を渡す
        tempo, y_mono, sr, beat_times = extract_bpm_and_beats_from_loaded(y, sr, channel=channel)
        if beat_times is None or len(beat_times) < 4:
            st.error("拍が十分に検出できませんでした。別の音源を試すか、拍数を手動で調整してください。")
            raise RuntimeError("insufficient_beats")

        st.success(f"検出されたテンポ：{round(tempo)} BPM")

        beats_per_bar = st.number_input("1小節の拍数", min_value=1, max_value=12, value=4)

        # ズレを補正
        bar_segments = get_bar_segments(beat_times, beats_per_bar)
        corrected_bar = get_corrected_bar_times(bar_segments, tempo, beats_per_bar)

        # apply correction on the mono signal used for beat detection
        corrected_audio = apply_timing_correction(y_mono, sr, bar_segments, corrected_bar)

        if corrected_audio is None or corrected_audio.size == 0:
            st.error("補正後の音声を生成できませんでした。")
            raise RuntimeError("empty_corrected_audio")

        # 出力前にピーク正規化（クリッピング回避）
        peak = np.max(np.abs(corrected_audio)) if isinstance(corrected_audio, np.ndarray) else None
        if peak and peak > 0:
            corrected_audio = corrected_audio / peak * 0.99

        # 補正済み音源を保存
        corrected_buffer = BytesIO()
        sf.write(corrected_buffer, corrected_audio, sr, format='WAV')

        # ダウンロードボタン
        st.download_button("📥 補正済み音源をダウンロード", corrected_buffer.getvalue(), "corrected.wav", "audio/wav")

        tab1, tab2 = st.tabs(["元音源",  "補正済み"])
        with tab1:
             st.audio(audio_bytes, format='audio/wav')
        with tab2:
            corrected_buffer.seek(0)
            st.audio(corrected_buffer.read(), format='audio/wav')

        bar_bpms = estimate_bar_bpms(bar_segments)
        st.subheader("小節ごとのテンポ推定")
        st.dataframe(bar_bpms)

        corrected_segments = get_corrected_bar_segments(corrected_bar)
        corrected_bpms = estimate_corrected_bpms(corrected_segments, beats_per_bar)
        st.subheader("補正後の小節ごとのテンポ推定")
        st.dataframe(corrected_bpms)

        plot_bpms(bar_bpms, title="補正前のテンポ変化", column="推定BPM")
        plot_bpms(corrected_bpms, title="補正後のテンポ変化", column="補正後BPM")
    except RuntimeError:
        # 意図的な中断（ユーザ向けメッセージは既に表示済み）
        pass
    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
        raise