"""MoviePy audio reading with ffmpeg."""

import subprocess as sp

import numpy as np

from moviepy.config import FFMPEG_BINARY
from moviepy.tools import cross_platform_popen_params, ffmpeg_escape_filename
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos


class FFMPEG_AudioReader:
    """A class to read the audio in either video files or audio files
    using ffmpeg. ffmpeg will read any audio and transform them into
    raw data.

    Parameters
    ----------

    filename
      Name of any video or audio file, like ``video.mp4`` or
      ``sound.wav`` etc.

    buffersize
      The size of the buffer to use. Should be bigger than the buffer
      used by ``write_audiofile``

    print_infos
      Print the ffmpeg infos on the file being read (for debugging)

    fps
      Desired frames per second in the decoded signal that will be
      received from ffmpeg

    nbytes
      Desired number of bytes (1,2,4) in the signal that will be
      received from ffmpeg

    audio_stream_index
      The index of the audio stream to read from the file.
    """

    def __init__(
        self,
        filename,
        buffersize,
        decode_file=False,
        print_infos=False,
        fps=44100,
        nbytes=2,
        nchannels=2,
        audio_stream_index=0,
    ):
        # TODO bring FFMPEG_AudioReader more in line with FFMPEG_VideoReader
        # E.g. here self.pos is still 1-indexed.
        # (or have them inherit from a shared parent class)
        self.filename = filename
        self.nbytes = nbytes
        self.fps = fps
        self.format = "s%dle" % (8 * nbytes)
        self.codec = "pcm_s%dle" % (8 * nbytes)
        self.nchannels = nchannels
        infos = ffmpeg_parse_infos(
            filename, decode_file=decode_file, print_infos=print_infos
        )
        self.duration = infos["duration"]
        self.bitrate = infos["audio_bitrate"]
        self.infos = infos
        self.proc = None
        self.audio_stream_index = audio_stream_index

        self.n_frames = int(self.fps * self.duration)
        self.buffersize = min(self.n_frames + 1, buffersize)
        self.buffer = None
        self.buffer_startframe = 1
        self.initialize()
        self.buffer_around(1)

    def initialize(self, start_time=0):
        """Opens the file, creates the pipe."""
        self.close()  # if any

        i_arg = ["-i", ffmpeg_escape_filename(self.filename)]
        # Keep the stream selection and the seek on the output side.
        # An input-side seek (``-ss`` before ``-i``) can start the raw audio
        # stream one sample off the requested position, which would make
        # backward jumps read the wrong samples. The output-side seek is
        # sample-accurate and always reselects the requested audio stream.
        stream_args = ["-vn", "-map", "0:a:%d" % self.audio_stream_index]
        if start_time != 0:
            stream_args += ["-ss", "%.05f" % start_time]

        cmd = (
            [FFMPEG_BINARY]
            + i_arg
            + stream_args
            + [
                "-loglevel",
                "error",
                "-f",
                self.format,
                "-acodec",
                self.codec,
                "-ar",
                "%d" % self.fps,
                "-ac",
                "%d" % self.nchannels,
                "-",
            ]
        )

        popen_params = cross_platform_popen_params(
            {
                "bufsize": self.buffersize,
                "stdout": sp.PIPE,
                "stderr": sp.PIPE,
                "stdin": sp.DEVNULL,
            }
        )

        self.proc = sp.Popen(cmd, **popen_params)

        self.pos = np.round(self.fps * start_time)

    def skip_chunk(self, chunksize):
        """Skip a chunk of audio data by reading and discarding the specified number of
        frames from the audio stream. The audio stream is read from the `proc` stdout.
        After skipping the chunk, the `pos` attribute is updated accordingly.

        Parameters
        ----------
        chunksize (int):
          The number of audio frames to skip.
        """
        _ = self.proc.stdout.read(self.nchannels * chunksize * self.nbytes)
        self.proc.stdout.flush()
        self.pos = self.pos + chunksize

    def read_chunk(self, chunksize):
        """Read a chunk of audio data from the audio stream.

        This method reads a chunk of audio data from the audio stream. The
        specified number of frames, given by `chunksize`, is read from the
        `proc` stdout. The audio data is returned as a NumPy array, where
        each row corresponds to a frame and each column corresponds to a
        channel. If there is not enough audio left to read, the remaining
        portion is padded with zeros, ensuring that the returned array has
        the desired length. The `pos` attribute is updated accordingly.

        Parameters
        ----------
        chunksize (float):
          The desired number of audio frames to read.

        """
        # chunksize is not being autoconverted from float to int
        chunksize = int(round(chunksize))
        s = self.proc.stdout.read(self.nchannels * chunksize * self.nbytes)
        data_type = {1: "int8", 2: "int16", 4: "int32"}[self.nbytes]
        if hasattr(np, "frombuffer"):
            result = np.frombuffer(s, dtype=data_type)
        else:
            result = np.fromstring(s, dtype=data_type)
        result = (1.0 * result / 2 ** (8 * self.nbytes - 1)).reshape(
            (int(len(result) / self.nchannels), self.nchannels)
        )

        # Pad the read chunk with zeros when there isn't enough audio
        # left to read, so the buffer is always at full length.
        pad = np.zeros((chunksize - len(result), self.nchannels), dtype=result.dtype)
        result = np.concatenate([result, pad])
        # self.proc.stdout.flush()
        self.pos = self.pos + chunksize
        return result

    def seek(self, pos):
        """Read a frame at time t. Note for coders: getting an arbitrary
        frame in the video with ffmpeg can be painfully slow if some
        decoding has to be done. This function tries to avoid fectching
        arbitrary frames whenever possible, by moving between adjacent
        frames.
        """
        if (pos < self.pos) or (pos > (self.pos + 1000000)):
            t = 1.0 * pos / self.fps
            self.initialize(t)
        elif pos > self.pos:
            self.skip_chunk(pos - self.pos)
        # last case standing: pos = current pos
        self.pos = pos

    def get_frame(self, tt):
        """Retrieve the audio frame(s) corresponding to the given timestamp(s).

        Parameters
        ----------
        tt (float or numpy.ndarray):
          The timestamp(s) at which to retrieve the audio frame(s).
          If `tt` is a single float value, the frame corresponding to that
          timestamp is returned. If `tt` is a NumPy array of timestamps, an
          array of frames corresponding to each timestamp is returned.
          Timestamps outside of the file (negative or not smaller than the
          duration) yield silent frames.
        """
        if isinstance(tt, np.ndarray):
            tt = np.asarray(tt, dtype=float)

            result = np.zeros((len(tt), self.nchannels))

            # An empty query returns an empty array while keeping the number
            # of channels intact.
            if tt.size == 0:
                return result

            # The np.round in the next line is super-important.
            # Removing it results in artifacts in the noise.
            frames = np.round(self.fps * tt).astype(int)
            valid = (frames >= 0) & (frames < self.n_frames)

            # Every requested time out of range stays silent, even when all
            # of them are out of range.
            if not valid.any():
                return result

            valid_frames = frames[valid]

            # Look up each unique requested frame only once, in sorted order,
            # so forward and adjacent reads reuse the streaming buffer instead
            # of restarting ffmpeg per timestamp. The original (possibly
            # shuffled or duplicated) request order is restored afterwards.
            unique_frames = np.unique(valid_frames)
            frame_values = np.empty((len(unique_frames), self.nchannels), dtype=float)
            for index, frame_number in enumerate(unique_frames):
                if not (
                    0
                    <= (frame_number - self.buffer_startframe)
                    < len(self.buffer)
                ):
                    # out of the buffer: recenter the buffer
                    self.buffer_around(frame_number)
                frame_values[index] = self.buffer[
                    frame_number - self.buffer_startframe
                ]

            # Map the unique frames back onto each requested position so the
            # original (shuffled or duplicated) request order is preserved.
            valid_positions = np.flatnonzero(valid)
            result[valid_positions] = frame_values[
                np.searchsorted(unique_frames, valid_frames)
            ]
            return result
        else:
            # Use the same rounding rule as the array branch above.
            ind = int(round(self.fps * tt))
            if ind < 0 or ind >= self.n_frames:  # out of time: return 0
                return np.zeros(self.nchannels)

            if not (0 <= (ind - self.buffer_startframe) < len(self.buffer)):
                # out of the buffer: recenter the buffer
                self.buffer_around(ind)

            # read the frame in the buffer
            return self.buffer[ind - self.buffer_startframe]

    def buffer_around(self, frame_number):
        """Fill the buffer with frames, centered on frame_number if possible."""
        # start-frame for the buffer
        new_bufferstart = max(0, frame_number - self.buffersize // 2)

        if self.buffer is not None:
            current_f_end = self.buffer_startframe + self.buffersize
            if new_bufferstart < current_f_end < new_bufferstart + self.buffersize:
                # We already have part of what must be read
                conserved = current_f_end - new_bufferstart
                chunksize = self.buffersize - conserved
                array = self.read_chunk(chunksize)
                self.buffer = np.vstack([self.buffer[-conserved:], array])
            else:
                self.seek(new_bufferstart)
                self.buffer = self.read_chunk(self.buffersize)
        else:
            self.seek(new_bufferstart)
            self.buffer = self.read_chunk(self.buffersize)

        self.buffer_startframe = new_bufferstart

    def close(self):
        """Closes the reader, terminating the subprocess if is still alive."""
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.stdout.close()
                self.proc.stderr.close()
                self.proc.wait()
            self.proc = None

    def __del__(self):
        # If the garbage collector comes, make sure the subprocess is terminated.
        self.close()
