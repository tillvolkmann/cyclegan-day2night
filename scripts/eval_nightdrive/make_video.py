import os, sys
import subprocess
import re
import shutil
import pandas as pd

def ffmpeg_vstack(input0, input1, output, frame_rate):
    cmd = f"ffmpeg -r {str(frame_rate)} -i {input0} -i {input1} -filter_complex vstack=inputs=2 -vcodec libx264 -crf 18 {output}"
    subprocess.call(cmd, shell=True)


def ffmpeg_hstack(input0, input1, output, frame_rate):
    cmd = f"ffmpeg -r {str(frame_rate)} -i {input0} -i {input1} -filter_complex hstack=inputs=2 -vcodec libx264 -crf 18 {output}"
    subprocess.call(cmd, shell=True)

def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1
    out = subprocess.check_output(["ffprobe",filename,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=r_frame_rate"])
    rate = out.split('=')[1].strip()[1:-1].split('/')
    if len(rate)==1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1])
    return -1


if __name__ == "__main__":

    """
    Example call:
        cd ~/projects/git-forks/pytorch-CycleGAN-and-pix2pix/
        python3 ./scripts/eval_nightdrive/make_video.py
    """
    # Local specs
    which_host = 'dsr'  # Options: ['till', 'dsr']
    load_from = 'folder'  # 'bbdjson'  # Options: ['folder', 'bbdjson']

    # Set paths
    if which_host == 'till':
        video_dir = "/home/till/SharedFolder/CurrentDatasets/bdd100k/videos/train"
        out_dir = "/home/till/SharedFolder/CurrentDatasets/bdd100k_video_converted"
        json_path = "/home/till/SharedFolder/CurrentDatasets/bdd100k/videos/bdd100k_labels_videos.json"
        project_root = "/home/till/projects/git-forks/pytorch-CycleGAN-and-pix2pix/"
        project_nightdrive_root = "/home/SharedFolder/git/tillvolkmann/night-drive"
        gpu_ids = -1
    elif which_host == 'dsr':
        video_dir = "/home/SharedFolder/CurrentDatasets/bdd100k/videos/best"  # "/home/SharedFolder/CurrentDatasets/bdd100k/videos/train"
        out_dir = "/home/SharedFolder/CurrentDatasets/bdd100k_video_converted"
        project_root = "/home/SharedFolder/git/tillvolkmann/pytorch-CycleGAN-and-pix2pix/"
        json_path = "/home/SharedFolder/CurrentDatasets/bdd100k/videos/bdd100k_labels_videos.json"
        project_nightdrive_root = "/home/SharedFolder/git/tillvolkmann/night-drive"
        gpu_ids = 0
    else:
        raise Exception("Host not known.")

    # Specify model checkpoint to use for domain transfer
    name = 'cgan_aws_v032_backupbeforerestart'  # cgan_aws_v032_backupbeforerestart'  # name of model run / experiment
    epoch = 14
    sel_iter = 130000
    iter_not_epoch = False
    do_CAM = False
    do_timeofday_classify = True
    use_second_label_cam_blended = False

    # set frame rate (slightly accelerated will be good)
    frame_rate_other = 30
    frame_rate_blended = 60

    # set frames per window slide (approx 5 sec per slide will be good)
    fpc = frame_rate_blended * 10

    # estension of videos to be read
    video_extension = "mov"

    # suffix of domain-transformed frames
    suffix_transformed = "transfer_AtoB-"
    suffix_cam = "cam-"

    # Split all videos in file_basename and store frames in separate sub-directories
    os.chdir(project_root)
    if load_from == 'folder':
        files = [x for x in os.listdir(video_dir) if re.search(video_extension, x) is not None]
    elif load_from == 'bbdjson':
        df = pd.read_json(json_path)
        df.reset_index(drop=True, inplace=True)
        df = df.loc[df.attributes.apply(lambda x: x["timeofday"] == "daytime"), :]
        files = df.name.apply(lambda x: os.path.join(video_dir, x)).tolist()
    else:
        raise Exception("Source for loading not understood.")

    # modify out_dir by model run
    if not iter_not_epoch:
        out_dir = out_dir+"_"+name+"_e"+str(epoch)
    elif iter_not_epoch:
        out_dir = out_dir + "_" + name + "_i" + str(sel_iter)

    # run video maker
    n_files = len(files)
    for c, file_path in enumerate(files):

        # Write a status message
        print("\n\n\n\n\n\n")
        print("================================================================================")
        print(f"=== Processing file ({c} of {n_files}): {file_path}")
        print("================================================================================")

        # some files operations
        file = os.path.basename(file_path)  # strip off path
        file_basename, ext = os.path.splitext(file)

        # make temp dir for frame-by-frame processing, same name as file_basename
        if not iter_not_epoch:
            tmp_dir = os.path.join(video_dir, file_basename + "_" + name + "_e" + str(epoch))
        elif iter_not_epoch:
            tmp_dir = os.path.join(video_dir, file_basename + "_" + name + "_i" + str(sel_iter))
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        # split video
        print(f"\n\n--- Splitting video: {file} --------------------")
        cmd = f"ffmpeg -i {os.path.join(video_dir, file)} -start_number 0 {os.path.join(tmp_dir, 'frame-%d.png')}"
        subprocess.call(cmd, shell=True)

        # run GAN for domain transfer
        print(f"\n\n--- Transforming video: {file} --------------------")
        if not iter_not_epoch:
            subprocess.call(["python3", "./nightdrive_test.py",
                             "--model", "test",
                             "--direction", "AtoB",
                             "--phase", "test",
                             "--no_dropout",
                             "--preprocess", "none",
                             "--load_size", "1280",
                             "--gpu_ids", str(gpu_ids),
                             "--dataset_mode", "single",
                             "--out_style", "frames",
                             "--dataroot", tmp_dir,
                             "--results_dir", tmp_dir,
                             "--name", name,
                             "--epoch", str(epoch),
                             "--num_test", "-1",
                             "--norm", "instance",
                             "--batch_size", "32",
                             "--model_suffix", "_A"])
        elif iter_not_epoch:
            subprocess.call(["python3", "./nightdrive_test.py",
                             "--model", "test",
                             "--direction", "AtoB",
                             "--phase", "test",
                             "--no_dropout",
                             "--preprocess", "none",
                             "--load_size", "1280",
                             "--gpu_ids", str(gpu_ids),
                             "--dataset_mode", "single",
                             "--out_style", "frames",
                             "--dataroot", tmp_dir,
                             "--results_dir", tmp_dir,
                             "--name", name,
                             "--load_iter", str(sel_iter),
                             "--num_test", "-1",
                             "--norm", "instance",
                             "--batch_size", "32",
                             "--model_suffix", "_A"])


        # cam projection using fun from nightdrive
        print(f"\n\n--- Creating CAM overlays: {file} --------------------")
        if do_CAM or do_timeofday_classify:
            sys.path.append(project_nightdrive_root)
            path_weights = os.path.join(project_nightdrive_root,
                                        'classifier_timeofday/models/resnet18_timeofday_daynight_classifier_best.pth')
            path_fun = os.path.join(project_nightdrive_root,
                                    'classifier_timeofday/timeofday_cam.py')

            # cam with tod classify
            if do_CAM:
                # transformed frames
                cmd = f"python3 \
                    {path_fun} \
                    --path {tmp_dir} \
                    --suffix {suffix_transformed} \
                    --firstlabel \
                    --fontscale 1.8 \
                    --fontoutline 3 \
                    --weights {path_weights}"
                subprocess.call(cmd, shell=True)
                # original frames
                cmd = f"python3 \
                    {path_fun} \
                    --path {tmp_dir} \
                    --suffix '' \
                    --firstlabel \
                    --fontscale 1.8 \
                    --fontoutline 3 \
                    --weights {path_weights}"
                subprocess.call(cmd, shell=True)

            # timeofday classification without cam
            if do_timeofday_classify:
                # transformed frames
                cmd = f"python3 \
                    {path_fun} \
                    --path {tmp_dir} \
                    --suffix {suffix_transformed} \
                    --weights {path_weights} \
                    --classonly \
                    --firstlabel \
                    --fontscale 2.2 \
                    --fontoutline 4"
                subprocess.call(cmd, shell=True)
                # original frames
                cmd = f"python3 \
                    {path_fun} \
                    --path {tmp_dir} \
                    --suffix '' \
                    --weights {path_weights}  \
                    --classonly \
                    --firstlabel \
                    --fontscale 2.2 \
                    --fontoutline 4"
                subprocess.call(cmd, shell=True)


        # blend
        print(f"\n\n--- Blending video: {file} --------------------")
        subprocess.call(["python3", "./scripts/eval_nightdrive/frame_blender.py",
                         "--path", tmp_dir,
                         "--pattern1", "frame-",
                         "--pattern2", "frame-transfer_AtoB-",
                         "--out_basename", "frame-",
                         "--fpc", str(fpc),
                         "--blend"])

        # blend with cam
        if do_CAM or do_timeofday_classify:
            print(f"\n\n--- Blending video: {file} --------------------")
            if use_second_label_cam_blended:
                pattern2 = "frame-transfer_AtoB-cam-second-"
            else:
                pattern2 = "frame-transfer_AtoB-cam-"
            subprocess.call(["python3", "./scripts/eval_nightdrive/frame_blender.py",
                             "--path", tmp_dir,
                             "--pattern1", "frame-cam-",
                             "--pattern2", pattern2,
                             "--out_basename", "frame-cam-",
                             "--fpc", str(fpc),
                             "--blend"])

        # put videos together
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # get output file names
        print(f"\n\n--- Writing videos to {out_dir} --------------------")
        out_file_transfer = os.path.join(out_dir, file_basename + "-transfer_AtoB.mp4")
        out_file_blended = os.path.join(out_dir, file_basename + "-blended.mp4")
        out_file_hstack = os.path.join(out_dir, file_basename + "-hstack.mp4")
        out_file_vstack = os.path.join(out_dir, file_basename + "-vstack.mp4")
        if do_CAM or do_timeofday_classify:
            out_file_original_cam = os.path.join(out_dir, file_basename + "-cam.mp4")
            out_file_transfer_cam = os.path.join(out_dir, file_basename + "-transfer_AtoB-cam.mp4")
            out_file_hstack_cam = os.path.join(out_dir, file_basename + "-hstack-cam.mp4")
            out_file_vstack_cam = os.path.join(out_dir, file_basename + "-vstack-cam.mp4")
            out_file_blended_cam = os.path.join(out_dir, file_basename + "blended-cam.mp4")

        # compose blended video
        frame_pattern = os.path.join(tmp_dir, 'frame-blended-%d.png')
        cmd = f"ffmpeg -r {str(frame_rate_blended)} -f image2 -i {frame_pattern} -vcodec libx264 -crf 18 {out_file_blended}"
        subprocess.call(cmd, shell=True)
        # blend cam
        if do_CAM or do_timeofday_classify:
            frame_pattern = os.path.join(tmp_dir, 'frame-cam-blended-%d.png')
            cmd = f"ffmpeg -r {str(frame_rate_blended)} -f image2 -i {frame_pattern} -vcodec libx264 -crf 18 {out_file_blended_cam}"
            subprocess.call(cmd, shell=True)

        # compose fully-transformed video
        frame_pattern = os.path.join(tmp_dir, 'frame-transfer_AtoB-%d.png')
        cmd = f"ffmpeg -r {str(frame_rate_other)} -f image2 -i {frame_pattern} -vcodec libx264 -crf 18 {out_file_transfer}"
        subprocess.call(cmd, shell=True)

        # compose videos with cam overlay
        if do_CAM or do_timeofday_classify:
            # compose original video with cam overlay
            frame_pattern = os.path.join(tmp_dir, 'frame-cam-%d.png')
            cmd = f"ffmpeg -r {str(frame_rate_other)} -f image2 -i {frame_pattern} -vcodec libx264 -crf 18 {out_file_original_cam}"
            subprocess.call(cmd, shell=True)
            # compose fully-transferred video with cam overlay
            frame_pattern = os.path.join(tmp_dir, 'frame-transfer_AtoB-cam-%d.png')
            cmd = f"ffmpeg -r {str(frame_rate_other)} -f image2 -i {frame_pattern} -vcodec libx264 -crf 18 {out_file_transfer_cam}"
            subprocess.call(cmd, shell=True)

        # write vertically-concatenated video (original and fully-transformed)
        ffmpeg_vstack(file_path, out_file_transfer, out_file_vstack, frame_rate_other)
        # write horizontally-concatenated video (original and fully-transformed)
        ffmpeg_hstack(file_path, out_file_transfer, out_file_hstack, frame_rate_other)

        # write concatenated videos with cam overlay
        if do_CAM or do_timeofday_classify:
            # write horizontally-concatenated video (original and fully-transformed with cam overlay)
            ffmpeg_hstack(out_file_original_cam, out_file_transfer_cam, out_file_hstack_cam, frame_rate_other)
            # write vertically-concatenated video (original and fully-transformed with cam overlay)
            ffmpeg_vstack(out_file_original_cam, out_file_transfer_cam, out_file_vstack_cam, frame_rate_other)

        # remove temp dir
        shutil.rmtree(tmp_dir)


