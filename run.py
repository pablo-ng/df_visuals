import subprocess
import time

# TODO really need a resampler... from 44100 to 22050
# TODO drop shadow`?`

if subprocess.call("cargo build --release") != 0:
    exit()


input_device_index = 1
fps = 35
sample_rate = 48000
window_size = 2048
windowed = False
device_sample_rate = 48000
device_input_config = 0

particle_amp_threshold = 100.
particle_logo_probability = 0.2

for [artist_name, circle_radius, font_size] in [
    ["Tin &\nIron", 280., 180.],
    # ["Sereneti", 400., 200.],
    # ["Youphoria", 450., 200.],
    # ["3nlight", 340., 200.],
    # ["Abstract", 400., 200.],
    # ["Kevi Metal\nb2b Croney", 330., 130.],
]:
    p = subprocess.Popen(map(lambda val: str(val), [
        "./target/release/df_visuals.exe",
        input_device_index, artist_name,
        "--fps", fps,
        "-s", sample_rate,
        "-w", window_size,
        "--device-sample-rate", device_sample_rate,
        "--device-input-config", device_input_config,
        "--particle-amp-threshold", particle_amp_threshold,
        "--particle-logo-probability", particle_logo_probability,
        "--circle-radius", circle_radius,
        "--font-size", font_size,
    ] + (["--windowed"] if windowed else [])))
    time.sleep(20)
    p.terminate()
