import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import time
import threading
from timeit import timeit

n = 8
omega = 1100
N = 256

# Generating random signal.
x = np.zeros(N)

for i in range(1, n+1):
    A = np.random.random()
    phi = np.random.random()

    for t in range(N):
        x[t] += A * np.sin(omega/i * (t + 1) + phi)

plt.figure(figsize=(14, 8))
plt.title("Pseudo-random variable")
plt.plot(range(N), x, "r")  # random variable
plt.show()


# Simple DFT. (for comparison)
def discrete_fourier_transform(signal):
    signal = np.asarray(signal, dtype=float)
    N = signal.shape[0]

    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    # print(np.dot(M, signal))
    return np.dot(M, signal)



class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return



# Fast DFT wtith threads that creates only once.
def fast_fourier_transform_with_threads(signal, is_first=True):
    signal = np.asarray(signal, dtype=float)
    N = signal.shape[0]

    if N <= 2:
        return discrete_fourier_transform(signal)
    else:
        if is_first:
            even_thread = ThreadWithReturnValue(
                target=fast_fourier_transform_with_threads, 
                args=(signal[::2], False))
            even_thread.start()
            odd_thread = ThreadWithReturnValue(
                target=fast_fourier_transform_with_threads, 
                args=(signal[1::2], False))
            odd_thread.start()
            signal_odd = odd_thread.join()
            signal_even = even_thread.join()
        else:
            signal_even = fast_fourier_transform_with_threads(signal[::2], False)
            signal_odd = fast_fourier_transform_with_threads(signal[1::2], False)

        terms = np.exp(-2j * np.pi * np.arange(N) / N)

        
        return np.concatenate([signal_even + terms[:int(N / 2)] * signal_odd,
                               signal_even + terms[int(N / 2):] * signal_odd])

# Fast DFT wtith threads that creates on each recusive call.
def fast_fourier_transform_with_recursive_threads(signal):
    signal = np.asarray(signal, dtype=float)
    N = signal.shape[0]

    if N <= 2:
        return discrete_fourier_transform(signal)
    else:
        even_thread = ThreadWithReturnValue(
            target=fast_fourier_transform_with_recursive_threads, 
            args=(signal[::2], ))
        even_thread.start()
        odd_thread = ThreadWithReturnValue(
            target=fast_fourier_transform_with_recursive_threads, 
            args=(signal[1::2], ))
        odd_thread.start()

        signal_even = even_thread.join()
        signal_odd = odd_thread.join()
        terms = np.exp(-2j * np.pi * np.arange(N) / N)

        
        return np.concatenate([signal_even + terms[:int(N / 2)] * signal_odd,
                               signal_even + terms[int(N / 2):] * signal_odd])

# Fast DFT 
def fast_fourier_transform(signal):
    signal = np.asarray(signal, dtype=float)
    N = signal.shape[0]

    if N <= 2:
        return discrete_fourier_transform(signal)
    else:
        signal_even = fast_fourier_transform(signal[::2])
        signal_odd = fast_fourier_transform(signal[1::2])

        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([signal_even + terms[:int(N / 2)] * signal_odd,
                               signal_even + terms[int(N / 2):] * signal_odd])


# Comparing DFT implementation with FFT implementation.
DFT = discrete_fourier_transform(x)
DFT_R = DFT.real
DFT_I = DFT.imag

FFT = fast_fourier_transform(x)
FFT_R = FFT.real
FFT_I = FFT.imag
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,14))
ax1.plot(range(N), DFT_R)
ax2.plot(range(N), DFT_I)
ax3.plot(range(N), FFT_R)
ax4.plot(range(N), FFT_I)
plt.show()


number_of_tests = 5
fft_threads = list()
fft_recursive_threads = list()
fft = list()
sizes = range(2, 1024, 16)
for N in sizes:
    fft_threads.append(timeit(
        lambda: fast_fourier_transform_with_threads(x), number=number_of_tests))
    fft_recursive_threads.append(timeit(
        lambda: fast_fourier_transform_with_recursive_threads(x), number=number_of_tests))
    fft.append(timeit(
        lambda: fast_fourier_transform(x), number=number_of_tests))
    print('N is {} of 1024'.format(N))

plt.plot(sizes, fft_threads, label='FFT with threads')
plt.plot(sizes, fft_recursive_threads, label='FFT with recusive threads')
plt.plot(sizes, fft, label='FFT')
plt.xlabel('N')
plt.ylabel('time')
plt.legend()
plt.show()
