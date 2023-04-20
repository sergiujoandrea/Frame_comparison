# def video_size(filename):
#     # Open the video file
#     video = cv2.VideoCapture(filename)
#
#     # Get the total number of frames in the video
#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # Get the size of each frame in the video
#     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # Calculate the size of the video
#     video_size = (width * height) * total_frames
#
#     # Print the size of the video in bytes
#     print(f"Video size: {video_size} bytes")
#
#     # Release the video
#     video.release()




#      def quality_image(filename):
#     # Use a breakpoint in the code line below to debug your script.
#     # Load original and compressed videos
#     cap_original = cv2.VideoCapture('videos/sample_1280x720.avi')
#     cap_compressed = cv2.VideoCapture(filename)
#
#     # Extract frames from videos
#     ret, frame_original = cap_original.read()
#     ret, frame_compressed = cap_compressed.read()
#
#     # Convert frames to grayscale
#     frame_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
#     frame_compressed = cv2.cvtColor(frame_compressed, cv2.COLOR_BGR2GRAY)
#
#     # Calculate PSNR
#     psnr = peak_signal_noise_ratio(frame_original, frame_compressed)
#
#     # Calculate SSIM
#     ssim = structural_similarity(frame_original, frame_compressed)
#
#     print(f'PSNR: {psnr:.2f} dB')
#     print(f'SSIM: {ssim:.2f}')
#     cap_original.release()
#     cap_compressed.release()
#
# # Press the green button in the gutter to run the script.
# def psnr(filename):
#     video_file1 = 'videos/sample_1280x720.avi'
#     video_file2 = filename
#
#     cap1 = cv2.VideoCapture(video_file1)
#     cap2 = cv2.VideoCapture(video_file2)
#
#     psnr_values = []
#     ssim_values = []
#
#     while True:
#         ret1, frame1 = cap1.read()
#         ret2, frame2 = cap2.read()
#         if not (ret1 and ret2):
#             break
#
#         # Convert frames to grayscale for SSIM calculation
#         frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#         frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#
#         # Compute PSNR and SSIM for current frame pair
#         psnr = cv2.PSNR(frame1, frame2)
#         ssim = cv2.SSIM(frame1_gray, frame2_gray)
#
#         # Store PSNR and SSIM values
#         psnr_values.append(psnr)
#         ssim_values.append(ssim)
#
#     cap1.release()
#     cap2.release()
#
#     # Compute average PSNR and SSIM values for entire video pair
#     avg_psnr = sum(psnr_values) / len(psnr_values)
#     avg_ssim = sum(ssim_values) / len(ssim_values)
#
#     print("Average PSNR:", avg_psnr)
#     print("Average SSIM:", avg_ssim)

import cv2

# Load the avi and mpeg videos
avi = cv2.VideoCapture('video1.avi')
mpeg = cv2.VideoCapture('video2.mpeg')

# Get the video properties
avi_fps = avi.get(cv2.CAP_PROP_FPS)
mpeg_fps = mpeg.get(cv2.CAP_PROP_FPS)

# Set the frame counter to zero
frame_num = 0

# Loop over the frames in the videos
while True:
    # Read the next frame from each video
    avi_ret, avi_frame = avi.read()
    mpeg_ret, mpeg_frame = mpeg.read()

    # If either video has reached the end, break out of the loop
    if not avi_ret or not mpeg_ret:
        break

    # Convert the frames to grayscale
    avi_gray = cv2.cvtColor(avi_frame, cv2.COLOR_BGR2GRAY)
    mpeg_gray = cv2.cvtColor(mpeg_frame, cv2.COLOR_BGR2GRAY)

    # Compute the mean squared error (MSE) between the two frames
    mse = ((avi_gray.astype("float") - mpeg_gray.astype("float")) ** 2).mean()

    # Increment the frame counter
    frame_num += 1

# Release the video capture objects
avi.release()
mpeg.release()

# Compute the average MSE over all frames
avg_mse = mse / frame_num

# Print the results
print(f"Average MSE: {avg_mse}")
if avg_mse < 100:
    print("The image quality of the avi and mpeg videos is very similar")
else:
    print("The image quality of the avi and mpeg videos is different")