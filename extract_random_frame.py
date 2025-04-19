import cv2
import random
import argparse
import os

def extract_random_frame(video_path, output_path='frame.jpeg'):
    """
    Extracts a random frame from an MP4 video and saves it as JPEG.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output JPEG (default: 'frame.jpeg')
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Error: Could not determine frame count for {video_path}")
        cap.release()
        return False
    
    # Generate a random frame number
    random_frame = random.randint(0, total_frames - 1)
    
    # Set the video position to the random frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {random_frame} from {video_path}")
        cap.release()
        return False
    
    # Save the frame as JPEG
    cv2.imwrite(output_path, frame)
    
    # Release the video capture object
    cap.release()
    
    print(f"Successfully extracted frame {random_frame} of {total_frames} from {video_path}")
    print(f"Saved as {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract a random frame from a video file')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output', '-o', type=str, default='frame.jpeg', 
                        help='Path to save the output JPEG (default: frame.jpeg)')
    
    args = parser.parse_args()
    
    # Verify the video file exists
    if not os.path.isfile(args.video_path):
        print(f"Error: Video file {args.video_path} does not exist")
        return
    
    # Extract the random frame
    extract_random_frame(args.video_path, args.output)

if __name__ == "__main__":
    main()