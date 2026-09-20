"""Write a one-second 440 Hz tone as a 16 kHz mono WAV file using only the standard library.

Usage: python3 examples/make_wav.py [output.wav]
"""

import math
import struct
import sys
import wave

SAMPLE_RATE = 16000
SECONDS = 1
FREQUENCY_HZ = 440
AMPLITUDE = 12000


def main() -> None:
    output = sys.argv[1] if len(sys.argv) > 1 else "examples/tone.wav"
    frames = b"".join(
        struct.pack("<h", int(AMPLITUDE * math.sin(2 * math.pi * FREQUENCY_HZ * i / SAMPLE_RATE)))
        for i in range(SAMPLE_RATE * SECONDS)
    )
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(frames)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
