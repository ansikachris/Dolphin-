import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import random

class DolphinSonarGenerator:
    def __init__(self, sample_rate=192000):  # Dolphins use ultrasonic frequencies
        self.sample_rate = sample_rate
        
    def generate_echo_dataset(self, n_samples=1000, scenario='mixed'):
        """
        Generate synthetic dolphin sonar echo data
        
        Parameters:
        - n_samples: number of echo samples to generate
        - scenario: 'hunting', 'navigation', 'social', or 'mixed'
        """
        data = []
        
        for i in range(n_samples):
            # Generate base parameters based on scenario
            if scenario == 'hunting':
                distance = np.random.exponential(scale=3.0)  # Close-range hunting
                object_size = np.random.uniform(0.1, 2.0)    # Fish-sized objects
            elif scenario == 'navigation':
                distance = np.random.uniform(2.0, 50.0)      # Medium to long range
                object_size = np.random.uniform(1.0, 20.0)   # Rocks, coral, etc.
            elif scenario == 'social':
                distance = np.random.uniform(0.5, 10.0)      # Close to other dolphins
                object_size = np.random.uniform(1.5, 3.0)    # Dolphin-sized
            else:  # mixed
                distance = np.random.exponential(scale=5.0)
                object_size = np.random.uniform(0.1, 20.0)
            
            # Ensure minimum distance for realism
            distance = max(0.1, min(distance, 100.0))
            
            echo_data = self._generate_single_echo(distance, object_size)
            data.append(echo_data)
        
        return pd.DataFrame(data)
    
    def _generate_single_echo(self, distance, object_size):
        """Generate a single echo with realistic dolphin sonar characteristics"""
        
        # 1. DISTANCE TO OBJECT (meters)
        # Already provided as input parameter
        
        # 2. TIME DELAY (seconds)
        # Sound travels ~1500 m/s in water, round trip = 2x distance
        speed_of_sound = np.random.normal(1500, 50)  # Varies with temperature/salinity
        time_delay = (2 * distance) / speed_of_sound
        
        # 3. REFLECTION STRENGTH (intensity)
        # Based on object size, material properties, and distance
        base_reflection = self._calculate_reflection_strength(distance, object_size)
        
        # Add material property variation (soft tissue vs hard objects)
        material_factor = np.random.uniform(0.3, 1.0)  # 0.3 for soft, 1.0 for hard
        reflection_strength = base_reflection * material_factor
        
        # 4. FREQUENCY MODULATION
        # Dolphins use frequency sweeps (chirps) typically 40-150 kHz
        center_freq = np.random.uniform(60000, 120000)  # Hz
        bandwidth = np.random.uniform(20000, 60000)     # Hz
        sweep_rate = np.random.uniform(1000, 5000)      # Hz/ms
        
        # Frequency shift due to Doppler effect (if objects are moving)
        doppler_shift = np.random.normal(0, 100)  # Small random shifts
        
        # 5. SIGNAL CLARITY (noise level)
        # Signal-to-noise ratio depends on distance and environmental conditions
        base_snr = 20 - (distance * 0.5)  # SNR decreases with distance
        environmental_noise = np.random.uniform(0.5, 2.0)  # Ocean ambient noise
        signal_clarity = max(1.0, base_snr - environmental_noise)
        
        # ADDITIONAL REALISTIC PARAMETERS
        
        # Beam pattern effects
        beam_angle = np.random.uniform(-30, 30)  # degrees off-axis
        beam_attenuation = np.cos(np.radians(beam_angle)) ** 2
        
        # Multi-path reflections (simplified)
        multipath_delay = np.random.exponential(scale=0.001) if distance > 10 else 0
        multipath_strength = reflection_strength * 0.3 * np.random.random()
        
        # Absorption losses (frequency dependent)
        absorption_loss = self._calculate_absorption_loss(distance, center_freq)
        
        return {
            'distance_m': round(distance, 3),
            'time_delay_s': round(time_delay, 6),
            'reflection_strength_db': round(reflection_strength - absorption_loss, 2),
            'center_frequency_hz': int(center_freq + doppler_shift),
            'bandwidth_hz': int(bandwidth),
            'sweep_rate_hz_per_ms': round(sweep_rate, 1),
            'signal_clarity_snr_db': round(signal_clarity, 2),
            'beam_angle_deg': round(beam_angle, 1),
            'beam_attenuation_factor': round(beam_attenuation, 3),
            'multipath_delay_s': round(multipath_delay, 6),
            'multipath_strength_db': round(multipath_strength, 2),
            'object_size_m': round(object_size, 2),
            'doppler_shift_hz': round(doppler_shift, 1),
            'absorption_loss_db': round(absorption_loss, 2)
        }
    
    def _calculate_reflection_strength(self, distance, object_size):
        """Calculate reflection strength based on distance and object size"""
        # Simplified sonar equation: RL = SL - 2*TL + TS
        # where RL=received level, SL=source level, TL=transmission loss, TS=target strength
        
        source_level = 220  # dB re 1 μPa @ 1m (typical dolphin click)
        
        # Transmission loss (spherical spreading + absorption)
        transmission_loss = 20 * np.log10(distance) + 0.1 * distance
        
        # Target strength (depends on object size and shape)
        target_strength = 10 * np.log10(np.pi * (object_size/2)**2)  # Simplified sphere
        
        received_level = source_level - 2*transmission_loss + target_strength
        return received_level
    
    def _calculate_absorption_loss(self, distance, frequency):
        """Calculate frequency-dependent absorption loss in seawater"""
        # Simplified absorption coefficient for seawater
        freq_khz = frequency / 1000
        alpha = 0.1 * freq_khz**1.5  # dB/km (simplified)
        absorption_loss = alpha * (distance / 1000)  # Convert to km
        return absorption_loss
    
    def generate_time_series_echo(self, distance, object_size, duration=0.01):
        """Generate actual time-series waveform of an echo"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Generate click parameters
        echo_params = self._generate_single_echo(distance, object_size)
        
        # Create frequency sweep (chirp)
        f_start = echo_params['center_frequency_hz'] - echo_params['bandwidth_hz']/2
        f_end = echo_params['center_frequency_hz'] + echo_params['bandwidth_hz']/2
        
        # Direct path signal (outgoing click)
        click_duration = 0.0001  # 100 microseconds
        click_mask = t < click_duration
        outgoing_click = signal.chirp(t[click_mask], f_start, click_duration, f_end)
        
        # Echo signal (delayed and attenuated)
        delay_samples = int(echo_params['time_delay_s'] * self.sample_rate)
        echo_amplitude = 10**(echo_params['reflection_strength_db']/20) / 1000  # Scale down
        
        # Create complete signal
        total_signal = np.zeros(len(t))
        total_signal[click_mask] = outgoing_click
        
        if delay_samples < len(t):
            echo_end = min(delay_samples + len(outgoing_click), len(t))
            echo_length = echo_end - delay_samples
            total_signal[delay_samples:delay_samples + echo_length] += (
                outgoing_click[:echo_length] * echo_amplitude
            )
        
        # Add noise
        noise_level = 10**(-echo_params['signal_clarity_snr_db']/20)
        noise = np.random.normal(0, noise_level, len(total_signal))
        total_signal += noise
        
        return t, total_signal, echo_params

# Usage example
def main():
    generator = DolphinSonarGenerator()
    
    # Generate a dataset
    print("Generating dolphin sonar dataset...")
    df = generator.generate_echo_dataset(n_samples=500, scenario='mixed')
    
    print(f"Generated {len(df)} echo samples")
    print("\nDataset statistics:")
    print(df.describe())
    
    # Save to CSV
    df.to_csv('dolphin_sonar_data.csv', index=False)
    print("\nData saved to 'dolphin_sonar_data.csv'")
    
    # Generate and plot a sample time-series echo
    print("\nGenerating sample time-series echo...")
    t, waveform, params = generator.generate_time_series_echo(distance=5.0, object_size=1.0)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t * 1000, waveform)  # Convert to milliseconds
    plt.title('Dolphin Sonar Echo Waveform')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    # Spectrogram
    f, t_spec, Sxx = signal.spectrogram(waveform, generator.sample_rate, nperseg=1024)
    plt.pcolormesh(t_spec * 1000, f / 1000, 10 * np.log10(Sxx))
    plt.title('Spectrogram')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (kHz)')
    plt.colorbar(label='Power (dB)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSample echo parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
