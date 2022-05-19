import os
import sys
import cv2
import tqdm
import time

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)



def detect_kws(frame):
	kws = ['kill', 'slain', 'destroyed', 'stolen', 'dragon', 'baron', 'herald', 'blood', 'shut']
	result = ocr.ocr(frame, cls=True)
	detected = []
	for line in result:
		for kw in kws:
			if kw in line[1][0].strip().lower():
				detected.append(kw)	
	return detected


def extract_highlights(video_path, key_frames):
	key_frames.sort()
	segments = []
	start = -1
	end = -1
	for f in key_frames:
		if start == -1:
			start = f

		# if there is a gap of 10+ seconds between key moments, make it a new highlight
		elif f - start > 300:
			segments.append([start, end])
			start = f
			end = f

		else:
			end = f

	# add pad before and after highlight
	vid = cv2.VideoCapture(video_path)
	total_frames = vid.get(int(cv2.CAP_PROP_FRAME_COUNT))
	fps = vid.get(cv2.CAP_PROP_FPS)
	for i in range(len(segments)):
		segments[i] = (max(0, segments[i][0]-500), min(segments[i][1]+150, total_frames))

	# extract the frames from each highlight from the video
	highlight_frames = []
	
	frame_count = 0
	pbar = tqdm.tqdm(desc="Extracting Highlight Frames", total=int(total_frames))
	while True:
		retval, frame = vid.read()
		if frame is None:
			break

		for seg in segments:
			if frame_count in range(seg[0], seg[1]):
				highlight_frames.append(frame_count)
		frame_count += 1
		pbar.update(1)

	# return the extracted frames
	highlight_frames = list(set(highlight_frames))
	highlight_frames.sort()
	return highlight_frames


def write_highlight_video(frames, outpath, wh, video_path):
	w, h = wh
	invid = cv2.VideoCapture(video_path)
	fps = invid.get(cv2.CAP_PROP_FPS)
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	vid = cv2.VideoWriter(filename=outpath, fourcc=fourcc, fps=fps, frameSize=(int(invid.get(3)), int(invid.get(4))))
	frame_ind = 0
	frame_count = 0
	print("Writing video to:", outpath)
	while frame_ind < len(frames):
		retval, frame = invid.read()
		if frame_count == frames[frame_ind]:
			vid.write(frame)
			frame_ind += 1
		frame_count += 1
	vid.release()


def process_video(video_path, frame_skip=4):
	base = video_path.split('/')[-1].replace('.mp4', '')
	if not os.path.exists("results/{}".format(base)):
		os.makedirs("results/{}".format(base))
	vid = cv2.VideoCapture(video_path)
	frame_count = 0
	highlights = []
	frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
	pbar = tqdm.tqdm(desc="Extracting Highlights", total=int(frames))
	detect_time, detect_num = 0, 0
	start_time = time.perf_counter()
	while True:
		# enforce frame skip.
		retval, frame = vid.read()
		frame_count += 1
		pbar.update(1)
		if frame_count % frame_skip != 0:
			continue

		if frame is None:
			break
		
		# crop the frame to just the region displaying game-related text
		frame = frame[75:200, 300:1000]
		#cv2.imwrite("./frame_output/{}.png".format(frame_count), frame)

		# check for keywords on the frame
		time1 = time.perf_counter()
		kws = detect_kws(frame)
		for kw in kws:
			highlights.append([kw, frame_count])
		detect_num += 1
		detect_time += time.perf_counter()-time1
	print("\nAverage detect time per frame was {:.2f} seconds".format(detect_time/detect_num))

	# check if no highlights were detected
	if highlights == []:
		print("Did not detect any highlights. Stopping...")
		return

	# create a list of frames with keywords
	key_frames = []
	for h in highlights:
		key_frames.append(h[1])
	key_frames = list(set(key_frames))
	print("Detected {} key moments".format(len(key_frames)))

	# create highlight video using highlight_dict
	print("Extracting key moments from input video")
	highlight_frames = extract_highlights(video_path, key_frames)

	# write the highlight video
	print("Creating highlight video")
	prefix = "C:/Users/evani/Documents/Evan Programming/lol_highlight_generation/results/"
	write_highlight_video(highlight_frames, os.path.join(base, os.path.join(prefix+base+'/'+base+"_highlights.mp4")), [720, 1280], video_path)
	print("Finished in {:.2f} seconds".format(time.perf_counter()-start_time))


if __name__ == "__main__":
	base = "./video_input/"
	video_name = sys.argv[1]
	process_video(os.path.join(base, video_name), frame_skip=30)