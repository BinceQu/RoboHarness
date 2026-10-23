"""Optional head-camera recording, independent of observations and scoring."""
import json
from pathlib import Path
import time


def install(evaluator_cls):
    if getattr(evaluator_cls, '_roboharness_head_video', False):
        return
    original_start = evaluator_cls.start_recording
    original_write = evaluator_cls._write_video
    original_stop = evaluator_cls.stop_recording

    def report(self, closed=False):
        path = getattr(self, '_head_video_path', None)
        if path is None:
            return
        data = {'path': str(path), 'frames': self._head_video_frames,
                'width': getattr(self, '_head_video_width', None),
                'height': getattr(self, '_head_video_height', None),
                'fps': self._video_rate, 'closed': closed, 'updated_at': time.time()}
        target = path.with_suffix('.json')
        temp = target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(data, indent=2) + '\n')
        temp.replace(target)

    def close(self):
        writer = getattr(self, '_head_video_writer', None)
        if writer is not None:
            container, stream = writer
            try:
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
                self._head_video_writer = None
        report(self, closed=True)

    def start(self, fpath, rate=30):
        close(self)
        result = original_start(self, fpath, rate)
        self._head_video_path = Path(fpath).parent.parent / 'videos_head' / Path(fpath).name
        self._head_video_path.parent.mkdir(parents=True, exist_ok=True)
        self._head_video_frames = 0
        self._head_video_writer = None
        report(self)
        return result

    def write(self):
        result = original_write(self)
        if getattr(self, '_head_video_path', None) is None:
            return result
        key = self.robot_camera_names.get('head', '') + '::rgb'
        if key not in self.obs:
            return result
        import av
        import numpy as np
        frame = self.obs[key]
        if hasattr(frame, 'detach'):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        if self._head_video_writer is None:
            # Each keyframe closes a fragment, so a killed experiment retains
            # playable frames without waiting for the final MP4 index.
            container = av.open(str(self._head_video_path), mode='w', options={
                'movflags': 'frag_keyframe+empty_moov+default_base_moof', 'flush_packets': '1'})
            stream = container.add_stream('libx264', rate=self._video_rate)
            stream.height, stream.width = frame.shape[:2]
            stream.pix_fmt = 'yuv420p'
            stream.options = {'preset': 'veryfast', 'crf': '18', 'tune': 'zerolatency'}
            stream.codec_context.gop_size = self._video_rate
            stream.codec_context.max_b_frames = 0
            self._head_video_writer = (container, stream)
            self._head_video_height, self._head_video_width = frame.shape[:2]
        container, stream = self._head_video_writer
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
            container.mux(packet)
        self._head_video_frames += 1
        if self._head_video_frames == 1 or self._head_video_frames % self._video_rate == 0:
            report(self)
        return result

    def stop(self):
        try:
            close(self)
        finally:
            self._head_video_path = None
            original_stop(self)

    evaluator_cls.start_recording = start
    evaluator_cls._write_video = write
    evaluator_cls.stop_recording = stop
    evaluator_cls._roboharness_head_video = True
