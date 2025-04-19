from moviepy.video.io.VideoFileClip import VideoFileClip

def resize_video(input_path, output_path):
    # Load the video clip
    clip = VideoFileClip(input_path)
    
    # Resize to 540p (height = 540, maintaining aspect ratio)
    clip_resized = clip.resized(height=540)
    
    # Write the output file
    clip_resized.write_videofile(output_path, codec='libx264', fps=clip.fps)
    
    # Close the clip to free resources
    clip.close()
    clip_resized.close()

if __name__ == "__main__":
    input_video = "noshader_clip.mp4"  # Replace with your input file path
    output_video = "output_540p.mp4"  # Replace with your output file path
    resize_video(input_video, output_video)
