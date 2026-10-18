"""Independent PCM references shared by the resampler tests."""

import math
import struct


def sine_samples(rate: int, seconds: float, *, frequency: float = 1000.0) -> list[int]:
    """Generate a mono 16-bit sine at the given sample rate."""
    return [round(8000 * math.sin(2 * math.pi * frequency * index / rate)) for index in range(round(rate * seconds))]


def interleave_pcm(samples: list[int], channels: int = 1) -> bytes:
    """Interleave mono samples across channels and encode them as PCM16."""
    interleaved = [sample for sample in samples for _ in range(channels)]
    return struct.pack(f"<{len(interleaved)}h", *interleaved)


def reference_resample(samples: list[int], source_rate: int, target_rate: int = 24000) -> list[int]:
    """Interpolate a whole buffer at every output position, independently of the streaming resampler."""
    if not samples or source_rate == target_rate:
        return list(samples)

    # The number of output samples whose position is below the end of the input
    output_count = max(1, -(-len(samples) * target_rate // source_rate))
    resampled: list[int] = []
    for output_index in range(output_count):
        left_index, remainder = divmod(output_index * source_rate, target_rate)
        if left_index >= len(samples) - 1:
            resampled.append(samples[-1])
            continue

        # Round half to even exactly, rather than through a float
        exact = samples[left_index] * target_rate + (samples[left_index + 1] - samples[left_index]) * remainder
        quotient, rounding_remainder = divmod(exact, target_rate)
        if 2 * rounding_remainder > target_rate or (
            2 * rounding_remainder == target_rate and quotient % 2
        ):
            quotient += 1
        resampled.append(quotient)
    return resampled


def assert_samples_close(actual: list[int], expected: list[int]) -> None:
    """Compare samples, allowing the one LSB that float rounding can differ from exact rounding by."""
    assert len(actual) == len(expected)
    difference = max(
        (abs(actual_sample - expected_sample) for actual_sample, expected_sample in zip(actual, expected, strict=True)),
        default=0,
    )
    assert difference <= 1
