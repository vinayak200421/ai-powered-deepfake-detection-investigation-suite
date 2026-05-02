import numpy as np
import cv2
import torch
from src.pipeline import Pipeline, PipelineConfig
import gc

def main():
    print("Testing pipeline on random noise frames...")
    
    # Create 30 frames of 1280x720 random noise
    frames = []
    for _ in range(30):
        # Create a blank image with a "face" in the middle so MTCNN finds something
        # Wait, if MTCNN finds no face, it returns empty, which is also a valid test.
        # But let's create an actual valid video file so run_on_video works.
        pass

    # Actually let's just make a dummy video file
    out = cv2.VideoWriter("dummy.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
    for _ in range(30):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        # draw a simple face-like rectangle so mtcnn might trigger
        cv2.rectangle(frame, (600, 300), (680, 380), (200, 150, 150), -1)
        out.write(frame)
    out.release()
    
    print("Dummy video created. Running pipeline...")
    pipe = Pipeline(device="cuda")
    pipe.load_models()
    
    try:
        res = pipe.run_on_video("dummy.mp4", max_frames=30)
        print("Pipeline result:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
