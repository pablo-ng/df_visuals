import pygame
import random
import math
import numpy as np
import librosa
import pyaudio


# TODO should be overlapping windows


audio, sample_rate = librosa.load('music.mp3', sr=22050, mono=True, offset=15)

# initialize pygame
pygame.init()

# set up the display
WIDTH = 1066
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

artist_name = "Tin & Iron"

# set up the clock
clock = pygame.time.Clock()

# set up the font
initial_font_size = 64
font_size = initial_font_size
radius = 100
font = pygame.font.Font(None, initial_font_size)

# set up the particle list
particles = []

# set up the text
text = font.render(artist_name, True, (255, 255, 255))

# set up the text position
text_pos = [WIDTH / 2 - text.get_width() / 2, HEIGHT / 2 -
            text.get_height() / 2]

# set up the text rotation angle
text_angle = 0

center = np.array([WIDTH / 2, HEIGHT / 2])

# set up the FFT parameters
hop_length = 512
window_size = 2048

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1,
                rate=sample_rate, output=True)


def get_low_freq_amp(samples):
    dft = np.expand_dims(np.fft.fft(samples), 1)

    # compute the power spectrum
    power = np.abs(dft)**2

    # compute the frequency vector
    # freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # compute the mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=((window_size - 1) * 2), n_mels=128)

    # apply the mel filterbank to the power spectrum
    mel_power = np.reshape(np.dot(mel_basis, power), (-1,))

    # compute the log mel spectrogram
    log_mel_power = librosa.power_to_db(mel_power)

    # compute the low-frequency amplitude
    low_freq_amp = np.sum(log_mel_power[:16])

    return low_freq_amp


def get_particles(low_freq_amp, radius):
    global particles

    # create a new particle if the low-frequency amplitude is high enough
    if low_freq_amp > 100:
        theta = random.random() * 2 * math.pi
        # np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        velocity = np.array(
            [radius * math.cos(theta), radius * math.sin(theta)])
        position = center + velocity
        particle = {'position': position,
                    'velocity': 0.2 * velocity / np.linalg.norm(velocity),
                    'color': (random.randint(0, 100), random.randint(100, 255), random.randint(0, 100))}
        particles.append(particle)

    # update the position of each particle
    for particle in particles:
        particle['position'][0] += particle['velocity'][0] * \
            low_freq_amp / 10
        particle['position'][1] += particle['velocity'][1] * \
            low_freq_amp / 10

        # clip
        # TODO can particles be locked at center??
        norm = np.linalg.norm(particle['position'] - center)
        if norm < radius:
            particle['position'] = center + radius * \
                (particle['position'] - center) / norm

    # remove particles that are off the screen
    particles = [particle for particle in particles if particle['position'][0] >= 0 and particle['position']
                 [0] < WIDTH and particle['position'][1] >= 0 and particle['position'][1] < HEIGHT]

    return particles


samples = np.zeros((window_size,), dtype=np.float32)
i = 0
for sample in audio:

    stream.write(sample.tobytes())
    samples = np.roll(samples, 1)
    samples[0] = sample

    i = (i + 1) % hop_length

    if i == 0:

        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        # clear the screen
        screen.fill((5, 53, 23))

        low_freq_amp = get_low_freq_amp(samples)

        # compute the rotation angle of the text based on the low-frequency amplitude
        # text_angle = math.radians(low_freq_amp / 10)
        # text_angle = max(text_angle, math.radians(-20))
        # text_angle = min(text_angle, math.radians(20))
        # text_angle = math.degrees(text_angle)
        text_angle = 0

        # compute the font size based on the low-frequency amplitude
        k = 0.6
        font_size = max(24, int((1-k) * font_size +
                        k * (initial_font_size + low_freq_amp / 20)))
        radius = font_size + 60

        particles = get_particles(low_freq_amp, radius)

        # create the new font with the updated size
        font = pygame.font.Font(None, font_size)

        # render the text with the updated font and rotation angle
        text = font.render(artist_name, True, (255, 255, 255))
        text = pygame.transform.rotate(text, text_angle)

        # set up the text position
        text_pos = [WIDTH / 2 - text.get_width() / 2, HEIGHT / 2 -
                    text.get_height() / 2]

        pygame.draw.circle(screen, (3, 37, 23), center, radius)

        # draw the particles
        for particle in particles:
            pygame.draw.circle(screen, particle['color'], [int(
                particle['position'][0]), int(particle['position'][1])], 5)

        # make the drop-shadow
        text_bitmap = font.render(artist_name, True, (128, 128, 128))
        dropshadow_offset = 2
        screen.blit(
            text_bitmap, (text_pos[0]+dropshadow_offset, text_pos[1]+dropshadow_offset))

        # draw the text
        screen.blit(text, text_pos)

        # update the screen
        pygame.display.flip()

        # tick the clock
        # clock.tick(60)


stream.stop_stream()
stream.close()

p.terminate()

pygame.quit()
