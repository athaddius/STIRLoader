from pathlib import Path
import cv2
import numpy as np
import json
import torchvision.transforms as transforms
import torch
import logging
import tempfile
import subprocess as sp
import os


def getKfromcameramat(mat, scale=1.0):
    """Returns an intrinsic matrix, with possibility of scaling, in case we rescale images"""
    K = mat.astype(np.float32)
    K[0, 2] = K[0, 2] / scale
    K[1, 2] = K[1, 2] / scale
    K[0, 0] = K[0, 0] / scale
    K[1, 1] = K[1, 1] / scale
    return K


def getQ(baseline, K):
    """Gets Q backprojection matrix from K matrix and camera baseline"""

    Q = np.zeros((4, 4), np.float32)
    cx = K[0, 2]
    cy = K[1, 2]
    f = K[0, 0]
    Q[0, 3] = -cx
    Q[1, 3] = -cy
    Q[2, 3] = f
    Q[3, 2] = -1.0 / baseline
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    return Q


def getviddirs2d_STIR(datadir):
    """Grabs videos generated as clips from prepared STIR dataset"""
    datadir = Path(datadir)
    labdirs = list(datadir.glob("*"))

    expdirs = []
    for labdir in labdirs:
        expdirs.extend(list(labdir.glob("left*")))

    fulllist = set()
    for viddir in expdirs:
        seqdirs = viddir.glob("seq*")
        fulllist.update(seqdirs)
    fulllist = sorted(list(fulllist))
    return fulllist


def to_ori(x):
    """Converts from: a 0.0, 1.0 range tensor with shape [C, H, W]
    to: a 0, 255 range integer tensor, with shape [H, W, C]"""
    return (x.permute(1, 2, 0) * 255.0).byte()


def loadimcv(framename):
    """Returns frame in floating point rgb [H, W, C]"""
    frameim = cv2.imread(str(framename))
    frameim = cv2.cvtColor(frameim, cv2.COLOR_BGR2RGB)
    return (frameim / 255.0).astype(np.float32)


def rightnamefromleft(seqleft):
    """Replaces name of right folder with left
    Returns:
        rightseqpath: name for right vid
        vidname: name of left vid
        startname: starting path parts"""
    startname = seqleft.parts[:-2]
    vidname = seqleft.parts[-2]
    seqname = seqleft.parts[-1]
    rightvid = vidname.replace("left", "right", 1)
    rightseqpath = Path(*startname, rightvid, seqname)
    return rightseqpath, vidname, startname


class DataSequenceFull(torch.utils.data.IterableDataset):
    """Generates full sequences, returning an iterable dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.fullseq(withcal=True)

def getfile(basename):
    """Gets a single video"""


    datasets = []
    try:
        datasequence = STIRStereoClip(Path(basename))
        dataset = DataSequenceFull(datasequence)  # wraps in dataset
        datasets.append(dataset)
    except (AssertionError, IndexError) as e:
        logging.debug(
            f"error on {basename}: {e}, sometimes happens if depth not finished"
        )
        print(e)
    return datasets

def getclips(datadir="/data2/STIRDataset"):
    """Gets full length sequences from segmented ground truth data
    datadir: dataset directory for STIR dataset"""

    seqlist = getviddirs2d_STIR(datadir)  # list of seq<##> folders

    datasets = []
    for basename in seqlist:
        try:
            datasequence = STIRStereoClip(basename)
            dataset = DataSequenceFull(datasequence)  # wraps in dataset
            datasets.append(dataset)
        except (AssertionError, IndexError) as e:
            logging.debug(
                f"error on {basename}: {e}, sometimes happens if depth not finished"
            )
            print(e)
    return datasets


def filterlength(filename, numseconds, tofilter=False):
    """Throws indexerror if video length is over numseconds in length"""
    name = filename
    ms = name.split("ms-")
    starttime = int(ms[0])
    endtime = int(ms[1])
    duration = endtime - starttime
    if duration / 1000.0 > numseconds and tofilter:
        raise IndexError(f"video over {numseconds}s long, skipping")


class STIRStereoClip:
    """Loader for clip sequences
    takes in h264 video
    throws indexerror if no video
    """

    def __init__(self, leftseqpath, max_minutes=0.2):
        rightseqpath, vidname, startname = rightnamefromleft(leftseqpath)
        print(leftseqpath)
        self.leftbasename = leftseqpath  # seq01 file
        self.seqbase = Path(*leftseqpath.parts[0:-2])  # cuts off pieces /left/seq##
        withcal = True  # load calibration as well.
        calibfile = Path(self.seqbase, "calib.json")
        self.rightbasename = rightseqpath  # seq01 file
        vids_left = sorted(list(self.leftbasename.glob("frames/*.mp4")))
        vids_right = sorted(list(self.rightbasename.glob("frames/*.mp4")))
        if len(vids_right) == 0 or len(vids_left) == 0:
            raise IndexError(f"no videos in {leftseqpath}/frames")
        else:
            assert len(vids_left) == 1, "Number of left videos != 1"
            assert len(vids_right) == 1, "Number of right videos != 1"
        self.leftvidname = vids_left[0]
        filterlength(self.leftvidname.name, 60 * max_minutes)
        self.leftvidfolder = Path(*leftseqpath.parts[:-1])
        self.rightvidname = vids_right[0]
        self.rightvidfolder = Path(*rightseqpath.parts[:-1])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        logging.debug(f"STIRStereoClip: {self.leftvidname},{self.rightvidname}")
        self.basename = leftseqpath
        self.rightseqpath = rightseqpath
        self.vidfolder = Path(*self.basename.parts[:-1])
        if withcal:

            with open(calibfile, "r") as f:
                calib_dict = json.load(f)

            self.leftcameramat = np.array(calib_dict["leftcameramat"])
            self.rightcameramat = np.array(calib_dict["rightcameramat"])
            self.leftdistortioncoeffs = np.array(calib_dict["leftdistortioncoeffs"])
            self.rightdistortioncoeffs = np.array(calib_dict["rightdistortioncoeffs"])
            self.translation = np.array(calib_dict["translation"])
            self.rotation = np.array(calib_dict["rotation"])

            self.scale = 1.0
            left_cx = self.leftcameramat[0][2]
            right_cx = self.rightcameramat[0][2]
            self.disparitypad = (right_cx - left_cx) / self.scale
            assert np.all(
                self.rightdistortioncoeffs == 0
            ), "Need to add distortion math, not currently supported"
            self.baseline_mm = self.translation[0] * 1000.0
            self.K = getKfromcameramat(self.leftcameramat, self.scale)

            self.Q = getQ(self.baseline_mm, self.K)

    def getstartseg(self, left=True):
        """Returns segmentation image of start frame,
        left: whether to get segmentation for left frame (True: left, False: right)"""
        if left:
            base = self.basename
        else:
            base = self.rightseqpath
        start = Path(base, "segmentation", "icgstartseg.png")
        assert start.exists(), "Starting segmentation image doesn't exist"
        return loadimcv(start)

    def getendseg(self, left=True):
        """Returns segmentation image of end frame
        left: whether to get segmentation for left frame (True: left, False: right)"""
        if left:
            base = self.basename
        else:
            base = self.rightseqpath
        end = Path(base, "segmentation", "icgendseg.png")
        return loadimcv(end)

    def getstarticg(self, left=True):
        """Returns 'color'/IR image of start frame
        left: whether to get segmentation for left frame (True: left, False: right)"""
        if left:
            base = self.basename
        else:
            base = self.rightseqpath
        start = next(Path(base).glob("*_icgstart.png"))
        assert start.exists(), f"Start icg frame {start} doesn't exist"
        return loadimcv(start)

    def getendicg(self, left=True):
        """Returns 'color'/IR image of end frame
        left: whether to get segmentation for left frame (True: left, False: right)"""
        if left:
            base = self.basename
        else:
            base = self.rightseqpath
        end = next(Path(base).glob("*_icgend.png"))
        assert end.exists(), "end img doesn't exist"
        return loadimcv(end)

    def gettriple(self):
        """Returns image triple
        ir_im (from start of video), seg_im (from start of video), vis_im (RGB from from first frame)
        """
        im_seg = (cv2.cvtColor(self.getstartseg(), cv2.COLOR_BGR2GRAY) * 255.0).astype(
            np.uint8
        )
        im_vis = self.extractfirstframe()  # resized on extract
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)
        im_ir = cv2.cvtColor(self.getstarticg(), cv2.COLOR_RGB2BGR)
        return im_ir, im_seg, im_vis

    @staticmethod
    def getcentersfromseg(im_seg_float):
        """Grabs contour centers from a full resolution segmentation image.
        returns half-res center locations ***important to rescale image to display on
        Returns:
            centers: [[x, y], [x2, y2], ..., [xn, yn]] list of centers"""
        im_seg = (cv2.cvtColor(im_seg_float, cv2.COLOR_BGR2GRAY) * 255.0).astype(
            np.uint8
        )
        contours, hierarchy = cv2.findContours(
            im_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return []
            #raise IndexError(f"No contours were found in in image")
        centers = []  # set of bounding rectangle centers
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xcent = x + w // 2
            ycent = y + h // 2
            centers.append([xcent, ycent])
        return centers

    def getstartcenters(self, left=True):
        """Returns list of center points for the starting frame
        left: whether to get segmentation for left frame (True: left, False: right)
        Returns:
            centers: [[x, y], [x2, y2], ..., [xn, yn]] list of centers"""
        im_startseg_float = self.getstartseg(left)
        return self.getcentersfromseg(im_startseg_float)

    def getendcenters(self, left=True):
        """Returns list of center points for the ending frame
        left: whether to get segmentation for left frame (True: left, False: right)
        Returns:
            centers: [[x, y], [x2, y2], ..., [xn, yn]] list of centers"""
        im_endseg_float = self.getendseg(left)
        return self.getcentersfromseg(im_endseg_float)

    def getcenters(self):
        """Returns im_start and im_end with circles drawn on centers
        ir_im, seg_im, vis_im"""

        def drawcenters(im, centers):
            for pt in centers:
                im = cv2.circle(im, (pt[0], pt[1]), 6, (0, 0, 255), 2)
            return im

        im_vis = self.extractfirstframe()  # resized on extract
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)

        im_ir = cv2.cvtColor(self.getstarticg(), cv2.COLOR_RGB2BGR)
        im_startseg_float = self.getstartseg()
        startcenters = self.getcentersfromseg(im_startseg_float)
        im_ir_withcircles = drawcenters(im_ir, startcenters)

        im_ir_end = cv2.cvtColor(self.getendicg(), cv2.COLOR_RGB2BGR)
        im_endseg_float = self.getendseg()
        endcenters = self.getcentersfromseg(im_endseg_float)
        im_ir_end_withcircles = drawcenters(im_ir_end, endcenters)
        im_seg_float = cv2.cvtColor(im_startseg_float, cv2.COLOR_RGB2BGR)

        ## grab bb from startseg and first frame
        return cv2.hconcat([im_ir_withcircles, im_ir_end_withcircles])
        # return cv2.hconcat([im_vis, im_seg_float, im_ir_withcircles, im_ir_end_withcircles])

    def cross_correlation(self, patch1, patch2):
        """Calculates zero-mean ncc between two patches"""
        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            product /= stds
            return product

    def getsegsstereo(self, start=True):
        """From each left image, uses stereo to find ncc-closest patch along scanline for the right
        returns x, y, x2, y2 set of locations in images. y2=y"""
        if start:
            im_seg_float = self.getstartseg(left=True)
            im_seg_float_right = self.getstartseg(left=False)
            im_ir_left = self.getstarticg(left=True)
            im_ir_right = self.getstarticg(left=False)
        else:
            im_seg_float = self.getendseg(left=True)
            im_seg_float_right = self.getendseg(left=False)
            im_ir_left = self.getendicg(left=True)
            im_ir_right = self.getendicg(left=False)
        im_seg = (cv2.cvtColor(im_seg_float, cv2.COLOR_BGR2GRAY) * 255.0).astype(
            np.uint8
        )
        contours, hierarchy = cv2.findContours(
            im_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            pass
            #raise IndexError(f"no contours in im")

        centers = self.getcentersfromseg(im_seg_float_right)
        centerpairs = []
        centerpairsright = []
        for left_contour in contours:
            # show contour
            x, y, w, h = cv2.boundingRect(left_contour)
            cx1_unadjusted = x + w // 2
            cx1 = cx1_unadjusted + self.disparitypad
            cy1 = y + h // 2
            # print(f'{h}, {w}, {x}, {y}')
            left_patch = im_seg_float[y : y + h, x : x + w, :]
            left_patch_ir = im_ir_left[y : y + h, x : x + w, :]
            # print(f'{h}, {w}, {x}, {y}')
            otheropts = []
            otheropts_ir = []  # ir images of others
            ious = []
            nccs = []
            disps = []
            centers_matched = []
            for center in centers:
                cx2 = center[0]
                cy2 = center[1]
                disp = cx1 - cx2
                if (
                    abs(cy2 - cy1) > 10
                ):  # if difference is too large vertically, don't use it
                    continue
                if disp < 8 or disp > 105:  # likewise for disparities
                    continue
                # disparitymask the pieces that are out of bounds
                if cx2 - w // 2 < 0:
                    continue
                start = cx2 - w // 2
                end = start + w
                if end > 1280:
                    continue
                # print(f'{h}, {w}, {x}, {y}')
                right_patch = im_seg_float_right[y : y + h, start:end, :]
                right_patch_ir = im_ir_right[y : y + h, start:end, :]
                # print(f'{center}')
                right_patch_binary = right_patch.astype(np.uint8)
                left_patch_binary = left_patch.astype(np.uint8)
                # filter range
                otheropts.append(right_patch)
                otheropts_ir.append(right_patch_ir)
                # print(right_patch.shape)
                intersection = np.bitwise_and(
                    left_patch_binary, right_patch_binary
                ).sum()
                union = np.bitwise_or(left_patch_binary, right_patch_binary).sum()
                iou = intersection / union
                ious.append(iou)
                nccs.append(self.cross_correlation(left_patch_ir, right_patch_ir))
                disps.append(disp)
                centers_matched.append(cx2)
            # print(f'{center}')
            if len(ious) == 0:
                continue
            # print([o.shape for o in otheropts])
            iou_ims = [np.ones((h, w, 3)) * iou for iou in ious]

            ncc_ims = [np.ones((h, w, 3)) * x for x in nccs]
            showpatchmatches = False
            if showpatchmatches:
                cv2.imshow("allim", left_patch)
                cv2.imshow("allim_ir", cv2.cvtColor(left_patch_ir, cv2.COLOR_RGB2BGR))
                cv2.imshow("centerother", cv2.hconcat(otheropts))
                cv2.imshow(
                    "centerother_ir",
                    cv2.cvtColor(cv2.hconcat(otheropts_ir), cv2.COLOR_RGB2BGR),
                )
                cv2.imshow("centerscores", cv2.hconcat(iou_ims))
                cv2.imshow("centerscores_ncc", cv2.hconcat(ncc_ims))
                cv2.waitKey()

            metrics = nccs
            ind = np.argmin(metrics)
            disp = disps[ind]

            # get max disp
            # print(f'{disp}: disparity found')
            centerpairs.append([cx1_unadjusted, cy1])
            centerpairsright.append([centers_matched[ind], cy1])
        bothims = cv2.cvtColor(
            cv2.hconcat([im_ir_left, im_ir_right]), cv2.COLOR_RGB2BGR
        )
        _, imwidth, _ = im_ir_left.shape
        for matchl, matchr in zip(centerpairs, centerpairsright):
            x1, y1 = matchl
            x2, y2 = matchr
            x2 = x2 + imwidth
            cv2.line(bothims, (x1, y1), (x2, y2), (0, 165 / 255.0, 1.0), thickness=2)
        if False:
            cv2.imshow("matches", bothims)
            cv2.waitKey()
        return centerpairs, centerpairsright

    def get3DSegmentationPositions(self, start):
        """Gets positions of segmentation points in 3D by using getsegsstereo.
        start: whether to get starting or ending positions"""
        centerpairs, centerpairsright = np.array(self.getsegsstereo(start=start))
        unscaledK = getKfromcameramat(self.leftcameramat, 1.0)
        unscaleddisparity = self.disparitypad * self.scale
        unscaledQ = getQ(self.baseline_mm, unscaledK)
        disppoints = np.stack(
            (
                centerpairs[:, 0],
                centerpairs[:, 1],
                (centerpairs[:, 0] + unscaleddisparity) - centerpairsright[:, 0],
            ),
            axis=-1,
        )  # npts 3
        disp_homogeneous = np.pad(
            disppoints, ((0, 0), (0, 1)), "constant", constant_values=1
        )
        disp_homogeneous = disp_homogeneous @ unscaledQ.T
        disp_xyz = disp_homogeneous[:, :3] / disp_homogeneous[:, 3:4]
        # disp_homogeneous = disp_homogeneous @ unscaledQ.T
        # disp_xyz = disp_homogeneous[:,:2]#/disp_homogeneous[:,3:4]

        if False:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("TkAgg")
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.scatter(disp_xyz[:, 0], disp_xyz[:, 1], disp_xyz[:, 2])
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            ax.set_zlabel("Z Label")
            plt.show()
        return centerpairs, centerpairsright, disp_xyz

    def getrandompatchpair(self, segments=True):
        """Returns random patch from start
        ir_im, seg_im, vis_im
        if label is true, get a segmentation"""
        im_seg_float = self.getstartseg()
        im_seg = (cv2.cvtColor(im_seg_float, cv2.COLOR_BGR2GRAY) * 255.0).astype(
            np.uint8
        )
        contours, hierarchy = cv2.findContours(
            im_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            raise IndexError(f"no contours in im")
        if segments:
            randcontour = random.choice(contours)
            x, y, w, h = cv2.boundingRect(randcontour)
        else:
            x = np.random.randint(1280)
            y = np.random.randint(1024)
            w = 10
            h = 10
        im_vis = self.extractfirstframe()  # resized on extract
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)
        im_ir = cv2.cvtColor(self.getstarticg(), cv2.COLOR_RGB2BGR)
        im_seg_float = cv2.cvtColor(im_seg_float, cv2.COLOR_RGB2BGR)
        # im_seg_float = cv2.resize(im_seg_float, (640, 512))
        # im_ir = cv2.resize(im_ir, (640, 512))
        ## grab bb from startseg and first frame
        im_ir = cropbounds(im_ir, x, y, w, h)
        im_seg_float = cropbounds(im_seg_float, x, y, w, h)
        im_vis = cropbounds(im_vis, x, y, w, h)

        s = 22
        l = 21
        e = s + l
        assert e == 43
        im_ir_a = im_ir[s:e, s:e, :]
        im_ir_b = im_ir[e : e + l, e : e + l, :]
        im_vis_a = im_vis[s:e, s:e, :]
        im_vis_b = im_vis[e : e + l, e : e + l, :]
        im_seg_a = im_seg_float[s:e, s:e, :]
        im_seg_b = im_seg_float[e : e + l, e : e + l, :]
        return (
            im_vis_a,
            im_vis_b,
            cv2.hconcat([im_ir_a, im_vis_a, im_seg_a]),
            cv2.hconcat([im_ir_b, im_vis_b, im_seg_b]),
        )

    def getrandompatch(self, segments=True):
        """Returns random patch surrounding a segment from start_image
        returns concatenated patch: ir_im, seg_im, vis_im
        if segments is False, obtain a non-segment patch"""
        im_seg_float = self.getstartseg()
        im_seg = (cv2.cvtColor(im_seg_float, cv2.COLOR_BGR2GRAY) * 255.0).astype(
            np.uint8
        )
        contours, hierarchy = cv2.findContours(
            im_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            raise IndexError(f"no contours in im")
        if segments:
            randcontour = random.choice(contours)
            x, y, w, h = cv2.boundingRect(randcontour)
        else:
            x = np.random.randint(1280)
            y = np.random.randint(1024)
            w = 10
            h = 10
        im_vis = self.extractfirstframe()  # resized on extract
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)
        im_ir = cv2.cvtColor(self.getstarticg(), cv2.COLOR_RGB2BGR)
        im_seg_float = cv2.cvtColor(im_seg_float, cv2.COLOR_RGB2BGR)
        # im_seg_float = cv2.resize(im_seg_float, (640, 512))
        # im_ir = cv2.resize(im_ir, (640, 512))
        ## grab bb from startseg and first frame
        im_ir = cropbounds(im_ir, x, y, w, h)
        im_seg_float = cropbounds(im_seg_float, x, y, w, h)
        im_vis = cropbounds(im_vis, x, y, w, h)
        return cv2.hconcat([im_ir, im_vis, im_seg_float])

    def fullseq(self, withcal=True):
        """generator yields full sequence
        {ims, ims_right, ims_ori, ims_ori_right, xyzs, Ks, Qs, disparitypads}"""
        allframesleft, allframesright = self.extractallframes()
        # print(len(allframes))
        for frameleft, frameright in zip(allframesleft, allframesright):
            assert frameleft.shape == (1024, 1280, 3), "Frame size is not yet supported"
            assert frameright.shape == (
                1024,
                1280,
                3,
            ), "Frame size is not yet supported"
            image = self.transform(frameleft)
            image_right = self.transform(frameright)
            im_ori = to_ori(image)
            im_ori_right = to_ori(image_right)
            ims = [image]
            ims_right = [image_right]
            ims_ori = [im_ori]
            ims_ori_right = [im_ori_right]
            if withcal:
                K = torch.tensor([self.K])
                Q = torch.tensor([self.Q])
                disparitypad = torch.tensor([np.float32(self.disparitypad)])

                out = {
                    "ims": ims,
                    "ims_right": ims_right,
                    "ims_ori": ims_ori,
                    "ims_ori_right": ims_ori_right,
                    "Ks": K,
                    "Qs": Q,
                    "disparitypads": disparitypad,
                }
            else:
                out = {
                    "ims": ims,
                    "ims_right": ims_right,
                    "ims_ori": ims_ori,
                    "ims_ori_right": ims_ori_right,
                }
            yield out

    def extractallframes(self):
        """Extracts whole sequence into
        If SKIP is set in os.environ, this skips every SKIP frames

        usetmpdir:
            True extracts using tmpdir, and then loads the frames afterwards.
            False extracts using extractfullvideopipe"""

        def getframes(filename):
            size = (1280, 1024)
            usetmpdir = False
            if usetmpdir:  # complicated .
                with tempfile.TemporaryDirectory() as tmpdirname:
                    self.extractfullvideo(filename, tmpdirname, "visible")
                    framenames = sorted(os.listdir(tmpdirname))
                    vidframes = []
                    for frame in framenames:
                        frame = loadimcv(os.path.join(str(tmpdirname), frame))
                        # frame = cv2.resize(frame, (640, 512))
                        # cv2.imshow('test_cv', frame)
                        # cv2.waitKey(1)
                        vidframes.append(frame)
                    return vidframes
            else:
                frames = self.extractfullvideopipe(filename, "visible")
                return [cv2.resize(x, size) for x in frames]

        leftvidframes = getframes(self.leftvidname)
        rightvidframes = getframes(self.rightvidname)
        if "SKIP" in os.environ:
            SKIP = int(os.environ["SKIP"])
            print(f"using different skip factor of {SKIP}")
        else:
            SKIP = 1
        if len(leftvidframes) % SKIP == 1:
            leftvidframes = leftvidframes[::SKIP]
            rightvidframes = rightvidframes[::SKIP]
        else:
            leftvidframes = leftvidframes[::SKIP] + [leftvidframes[-1]]
            rightvidframes = rightvidframes[::SKIP] + [rightvidframes[-1]]
        return leftvidframes, rightvidframes

    @staticmethod
    def extractfullvideo(videoname, outdir, segname):
        """Extracts video to <outdir>/*_<segname>.png using ffmpeg"""
        outstr = f"{str(outdir)}/%06d_{segname}.png"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(videoname),
            outstr,
        ]
        process = sp.run(command)

    @staticmethod
    def extractfullvideopipe(videoname, segname):
        """Extracts video to image list in RGB format
        Returns:
            frames: list of frames"""
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(videoname),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
        pipe = sp.Popen(command, stdout=sp.PIPE)
        cnt = 0
        W = 1280
        H = 1024
        frames = []
        while True:
            cnt += 1
            raw_image = pipe.stdout.read(W * H * 3)
            image = np.fromstring(raw_image, dtype="uint8")
            if image.shape[0] == 0:
                break
            else:
                image = image.reshape((H, W, 3))
                frames.append(image)
            # cv2.imshow('test', image)
            # cv2.waitKey(1)
        pipe.stdout.close()
        pipe.wait()
        return frames

    @staticmethod
    def extractfirstframefromname(videoname, outdir, segname):
        """Extract first frame from left video, saving to <outdir>/*_<segname>.png"""
        outstr = f"{str(outdir)}/%06d_{segname}.png"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(videoname),
            "-vframes",
            "1",
            outstr,
        ]
        process = sp.run(command)

    def extractfirstframe(self):
        """Extracts first frame into tmpdir, and then loads said frame.
        Extracts from left video.

        Overcomplicated, but it works."""

        def getframes(filename):
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.extractfirstframefromname(filename, tmpdirname, "visible")
                framenames = sorted(os.listdir(tmpdirname))
                vidframes = []
                for frame in framenames:
                    frame = loadimcv(os.path.join(str(tmpdirname), frame))
                    # frame = cv2.resize(frame, (640, 512))
                    vidframes.append(frame)
                return vidframes

        leftvidframe = getframes(self.leftvidname)[0]
        return leftvidframe
