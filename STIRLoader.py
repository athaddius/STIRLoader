from pathlib import Path
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
import logging
import tempfile
import subprocess as sp
import os
def getviddirs2d_STIR(datadir):
    """ grabs videos generated as clips from prepared STIR dataset"""
    datadir = Path(datadir)
    labdirs = list(datadir.glob('*'))
    
    expdirs = []
    for labdir in labdirs:
        expdirs.extend(list(labdir.glob('left*')))

    fulllist = set()
    for viddir in expdirs:
        seqdirs = viddir.glob('seq*')
        fulllist.update(seqdirs)
    fulllist = sorted(list(fulllist))
    return fulllist

def to_ori(x):
    return (x.permute(1, 2, 0) * 255.0).byte() # 0, 255 range, h, w, c

def loadimcv(framename):
    """
    returns frame in floating point rgb"""
    frameim = cv2.imread(str(framename))
    frameim = cv2.cvtColor(frameim, cv2.COLOR_BGR2RGB)
    return (frameim/255.0).astype(np.float32)

def rightnamefromleft(seqleft):
    """ returns
    rightseqpath: name for right vid
    vidname: name of left vid
    startname: starting path parts"""
    startname = seqleft.parts[:-2]
    vidname = seqleft.parts[-2]
    seqname = seqleft.parts[-1]
    rightvid = vidname.replace('left', 'right', 1)
    rightseqpath = Path(*startname, rightvid, seqname)
    return rightseqpath, vidname, startname

class DataSequenceFull(torch.utils.data.IterableDataset):
    """ generates full sequences"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.fullseq(withcal=False)

def getclips(datadir = '/data2/STIRDataset'):
    """ Gets full length sequences from segmented ground truth data
    datadir: dataset directory for STIR dataset
"""

    seqlist = getviddirs2d_STIR(datadir) # list of seq## folders

    datasets = []
    for basename in seqlist:
        try: # FIXME what does this throw?
            datasequence = STIRStereoClip(basename)
            dataset = DataSequenceFull(datasequence) # wraps in dataset
            datasets.append(dataset)
        except (AssertionError, IndexError) as e:
            logging.debug(f'error on {basename}: {e}, sometimes happens if depth not finished')
            print(e)
    return datasets

def filterlength(filename, seconds):
    """ throws indexerror if length is over seconds in length"""
    name = filename
    ms = name.split('ms-')
    starttime = int(ms[0])
    endtime = int(ms[1])
    duration = endtime - starttime
    if duration / 1000. > seconds:
        raise IndexError(f'video over {seconds}s long, skipping')
            


class STIRStereoClip():
    """ Loader for sequences without depth frames, and extracted clips
    takes in h264 video
    class for each pregenerated l/r, clip
    throws indexerror if no video
    pregenerated sequence with video clips"""
    def __init__(self, leftseqpath):
        #fail if not all
        rightseqpath, vidname, startname = rightnamefromleft(leftseqpath)
        print(leftseqpath)
        self.leftbasename = leftseqpath #seq01 file
        self.rightbasename = rightseqpath #seq01 file
        vids_left = sorted(list(self.leftbasename.glob('frames/*.mp4')))
        vids_right = sorted(list(self.rightbasename.glob('frames/*.mp4')))
        if len(vids_right) == 0 or len(vids_left) == 0:
            raise IndexError(f'no videos in {leftseqpath}/frames')
        else:
            assert len(vids_left) == 1, "vids_left is not one"
            assert len(vids_right) == 1, "vids_right is not one"
        self.leftvidname = vids_left[0]
        filterlength(self.leftvidname.name, 60*10)
        self.leftvidfolder = Path(*leftseqpath.parts[:-1])
        self.rightvidname = vids_right[0]
        self.rightvidfolder = Path(*rightseqpath.parts[:-1])
        #breakpoint() # /data2/STIR/0/left/seq## is leftseqpath
        self.transform = transforms.Compose(
                [transforms.ToTensor(),
                ])

        logging.debug(f'{self.leftvidname},{self.rightvidname} frames: Uncounted length')
        self.basename = leftseqpath
        self.rightseqpath = rightseqpath
        self.vidfolder = Path(*self.basename.parts[:-1])

    def getstartseg(self, left=True):
        """ Returns segmentation image of start frame"""
        if left:
            start = Path(self.basename, 'segmentation', 'icgstartseg.png')
        else:
            start = Path(self.rightseqpath, 'segmentation', 'icgstartseg.png')
        assert start.exists(), "start image doesn't exist"
        return loadimcv(start)

    def getendseg(self, left=True):
        """ Returns segmentation image of end frame"""
        if left:
            end = Path(self.basename, 'segmentation', 'icgendseg.png')
        else:
            end = Path(self.rightseqpath, 'segmentation', 'icgendseg.png')
        return loadimcv(end)

    def getstarticg(self, left=True):
        """ Returns segmentation image of start frame"""
        if left:
            base = self.basename
        else:
            base = self.rightseqpath
        start = next(Path(base).glob('*_icgstart.png'))
        assert start.exists(), "start icg doesn't exist"
        return loadimcv(start)

    def getendicg(self, left=True):
        """ Returns segmentation image of end frame"""
        if left:
            base = self.basename
        else:
            base = self.rightseqpath
        end = next(Path(base).glob('*_icgend.png'))
        assert end.exists(), "end img doesn't exist"
        return loadimcv(end)

    def getrandomtriple(self):
        """ Returns random triple from start
        ir_im, seg_im, vis_im"""
        im_seg = (cv2.cvtColor(self.getstartseg(), cv2.COLOR_BGR2GRAY) *255.).astype(np.uint8)
        im_vis = self.extractfirstframe() # resized on extract
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)
        im_ir = cv2.cvtColor(self.getstarticg(), cv2.COLOR_RGB2BGR)
        im_seg = cv2.resize(im_seg, (640, 512))
        im_ir = cv2.resize(im_ir, (640, 512))
        ## grab bb from startseg and first frame
        return im_ir, im_seg, im_vis

    @staticmethod
    def getcentersfromseg(im_seg_float):
        """ grabs centers from full resolution segmentation image.
        returns half-res center locations ***important to rescale image to display on"""
        im_seg = (cv2.cvtColor(im_seg_float, cv2.COLOR_BGR2GRAY) *255.).astype(np.uint8)
        contours, hierarchy = cv2.findContours(im_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise IndexError(f'no contours in im')
        centers = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xcent = (x+w//2)
            ycent = (y+h//2)
            centers.append([xcent,ycent])
        return centers

    def getstartcenters(self, left=True):
        im_startseg_float = self.getstartseg(left)
        return self.getcentersfromseg(im_startseg_float)

    def getendcenters(self, left=True):
        # gets centers in downscaled format (//2)
        im_endseg_float = self.getendseg(left)
        return self.getcentersfromseg(im_endseg_float)

    def getcenters(self):
        """ Returns im_start and im_end with circles drawn on centers
        ir_im, seg_im, vis_im"""
        def drawcenters(im, centers):
            scale = 2
            for pt in centers:
                im = cv2.circle(im, (pt[0]//2, pt[1]//scale), 6, (0,0,255), 2)
            return im
             
        im_vis = self.extractfirstframe() # resized on extract
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)

        im_ir = cv2.cvtColor(self.getstarticg(), cv2.COLOR_RGB2BGR)
        im_ir = cv2.resize(im_ir, (640, 512))
        im_startseg_float = self.getstartseg()
        startcenters = self.getcentersfromseg(im_startseg_float)
        im_ir_withcircles = drawcenters(im_ir, startcenters)
        
        im_ir_end = cv2.cvtColor(self.getendicg(), cv2.COLOR_RGB2BGR)
        im_ir_end = cv2.resize(im_ir_end, (640, 512))
        im_endseg_float = self.getendseg()
        endcenters = self.getcentersfromseg(im_endseg_float)
        im_ir_end_withcircles = drawcenters(im_ir_end, endcenters)
        im_seg_float = cv2.cvtColor(im_startseg_float, cv2.COLOR_RGB2BGR)
        im_seg_float = cv2.resize(im_seg_float, (640, 512))
        
        ## grab bb from startseg and first frame
        return cv2.hconcat([im_ir_withcircles, im_ir_end_withcircles])
        #return cv2.hconcat([im_vis, im_seg_float, im_ir_withcircles, im_ir_end_withcircles])

    def cross_correlation(self, patch1, patch2):
        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            product /= stds
            return product

    def getsegsstereo(self, start=True):
        """ From each left image, uses stereo to find ncc-closest patch along scanline for the right
        returns x, y, x2, y set"""
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
        im_seg = (cv2.cvtColor(im_seg_float, cv2.COLOR_BGR2GRAY) *255.).astype(np.uint8)
        contours, hierarchy = cv2.findContours(im_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise IndexError(f'no contours in im')

        centers = self.getcentersfromseg(im_seg_float_right)
        centerpairs = []
        centerpairsright = []
        for left_contour in contours:
            # show contour
            x, y, w, h = cv2.boundingRect(left_contour)
            a=0
            cx1_unadjusted = x + w // 2
            cx1 = cx1_unadjusted + 90.6
            cy1 = y + h // 2
            x = x - a
            y = y - a
            w = w + 2*a
            h = h + 2*a
            #print(f'{h}, {w}, {x}, {y}')
            left_patch = im_seg_float[y:y+h, x:x+w,:]
            left_patch_ir = im_ir_left[y:y+h, x:x+w,:]
            #cv2.imshow("allim", left_patch)
            #cv2.imshow("allim_ir", cv2.cvtColor(left_patch_ir, cv2.COLOR_RGB2BGR))
            #print(f'{h}, {w}, {x}, {y}')
            otheropts = []
            otheropts_ir = [] # ir images of others
            ious = []
            nccs = []
            disps = []
            centers_matched = []
            for center in centers:
                cx2 = center[0]
                cy2 = center[1]
                disp = cx1 - cx2
                if abs(cy2 - cy1) >10:
                    continue
                if disp < 8 or disp > 105:
                    continue
                #disparitymask the pieces that are out of bounds
                if cx2-w//2 <0:
                    continue
                start = cx2-w//2
                if start + w > 1280:
                    continue
                end = start + w
                #print(f'{h}, {w}, {x}, {y}')
                right_patch = im_seg_float_right[y:y+h, start:end,:]
                right_patch_ir = im_ir_right[y:y+h, start:end,:]
                #print(f'{center}')
                right_patch_binary = right_patch.astype(np.uint8)
                left_patch_binary = left_patch.astype(np.uint8)
                # filter range
                otheropts.append(right_patch)
                otheropts_ir.append(right_patch_ir)
                #print(right_patch.shape)
                intersection = np.bitwise_and(left_patch_binary, right_patch_binary).sum()
                union = np.bitwise_or(left_patch_binary, right_patch_binary).sum()
                iou = intersection/union
                ious.append(iou)
                nccs.append(self.cross_correlation(left_patch_ir, right_patch_ir))
                disps.append(disp)
                centers_matched.append(cx2)
            #print(f'{center}')
            if len(ious) == 0:
                continue
            #cv2.imshow("centerother", cv2.hconcat(otheropts))
            #cv2.imshow("centerother_ir", cv2.cvtColor(cv2.hconcat(otheropts_ir), cv2.COLOR_RGB2BGR))
            #print([o.shape for o in otheropts])
            iou_ims = [np.ones((h, w, 3)) * iou for iou in ious]
            #cv2.imshow("centerscores", cv2.hconcat(iou_ims))

            ncc_ims = [np.ones((h, w, 3)) * x for x in nccs]
            #cv2.imshow("centerscores_ncc", cv2.hconcat(ncc_ims))

            metrics = nccs
            ind = np.argmin(metrics)
            disp = disps[ind]

            # get max disp
            #cv2.waitKey()
            #print(f'{disp}: disparity found')
            centerpairs.append([cx1_unadjusted, cy1])
            centerpairsright.append([centers_matched[ind], cy1])
        bothims = cv2.cvtColor(cv2.hconcat([im_ir_left, im_ir_right]), cv2.COLOR_RGB2BGR)
        _, imwidth, _ = im_ir_left.shape
        for matchl, matchr in zip(centerpairs, centerpairsright):
            x1, y1 = matchl
            x2, y2 = matchr
            x2 = x2 + imwidth
            cv2.line(bothims, (x1, y1), (x2, y2), (0, 165/255.0, 1.0), thickness=2)
        if False:
            cv2.imshow("matches", bothims)
            cv2.waitKey(1)
        return centerpairs, centerpairsright

    def getstartsegs3D(self, start):
        ## FIXME --show 3d positions
        centerpairs, centerpairsright = np.array(self.getsegsstereo(start=start))
        unscaledK = getKfromcameramat(self.leftcameramat, 1.0)
        unscaleddisparity = self.disparitypad * self.scale
        unscaledQ = Frame.getQ(self.baseline, unscaledK)
        disppoints = np.stack((centerpairs[:,0],centerpairs[:,1],(centerpairs[:,0]+unscaleddisparity)-centerpairsright[:,0]), axis=-1) # npts 3
        disp_homogeneous = np.pad(disppoints, ((0,0),(0, 1)),'constant', constant_values=1) # numpy pad
        disp_homogeneous = disp_homogeneous @ unscaledQ.T
        disp_xyz = disp_homogeneous[:,:3]/disp_homogeneous[:,3:4]
        #disp_homogeneous = disp_homogeneous @ unscaledQ.T
        #disp_xyz = disp_homogeneous[:,:2]#/disp_homogeneous[:,3:4]

        if False:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            ax.scatter(disp_xyz[:,0], disp_xyz[:,1], disp_xyz[:,2])
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()
        return centerpairs, centerpairsright, disp_xyz

        

    def getrandompatchpair(self, segments=True):
        """ Returns random patch from start
        ir_im, seg_im, vis_im
        if label is true, get a segmentation"""
        im_seg_float = self.getstartseg()
        im_seg = (cv2.cvtColor(im_seg_float, cv2.COLOR_BGR2GRAY) *255.).astype(np.uint8)
        contours, hierarchy = cv2.findContours(im_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise IndexError(f'no contours in im')
        if segments:
            randcontour = random.choice(contours)
            x, y, w, h = cv2.boundingRect(randcontour)
        else:
            x = np.random.randint(1280)
            y = np.random.randint(1024)
            w = 10
            h = 10
        im_vis = self.extractfirstframe() # resized on extract
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)
        im_ir = cv2.cvtColor(self.getstarticg(), cv2.COLOR_RGB2BGR)
        im_seg_float = cv2.cvtColor(im_seg_float, cv2.COLOR_RGB2BGR)
        im_seg_float = cv2.resize(im_seg_float, (640, 512))
        im_ir = cv2.resize(im_ir, (640, 512))
        ## grab bb from startseg and first frame
        im_ir = cropbounds(im_ir, x, y, w, h)
        im_seg_float = cropbounds(im_seg_float, x, y, w, h)
        im_vis = cropbounds(im_vis, x, y, w, h)

        s = 22
        l = 21
        e = s + l
        assert e == 43
        im_ir_a = im_ir[s:e, s:e, :]
        im_ir_b = im_ir[e:e+l, e:e+l, :]
        im_vis_a = im_vis[s:e, s:e, :]
        im_vis_b = im_vis[e:e+l, e:e+l, :]
        im_seg_a = im_seg_float[s:e, s:e, :]
        im_seg_b = im_seg_float[e:e+l, e:e+l, :]
        return im_vis_a, im_vis_b, cv2.hconcat([im_ir_a, im_vis_a, im_seg_a]), cv2.hconcat([im_ir_b, im_vis_b, im_seg_b])

    def getrandompatch(self, segments=True):
        """ Returns random patch from start
        ir_im, seg_im, vis_im
        if label is true, get a segmentation"""
        im_seg_float = self.getstartseg()
        im_seg = (cv2.cvtColor(im_seg_float, cv2.COLOR_BGR2GRAY) *255.).astype(np.uint8)
        contours, hierarchy = cv2.findContours(im_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise IndexError(f'no contours in im')
        if segments:
            randcontour = random.choice(contours)
            x, y, w, h = cv2.boundingRect(randcontour)
        else:
            x = np.random.randint(1280)
            y = np.random.randint(1024)
            w = 10
            h = 10
        im_vis = self.extractfirstframe() # resized on extract
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)
        im_ir = cv2.cvtColor(self.getstarticg(), cv2.COLOR_RGB2BGR)
        im_seg_float = cv2.cvtColor(im_seg_float, cv2.COLOR_RGB2BGR)
        im_seg_float = cv2.resize(im_seg_float, (640, 512))
        im_ir = cv2.resize(im_ir, (640, 512))
        ## grab bb from startseg and first frame
        im_ir = cropbounds(im_ir, x, y, w, h)
        im_seg_float = cropbounds(im_seg_float, x, y, w, h)
        im_vis = cropbounds(im_vis, x, y, w, h)
        return cv2.hconcat([im_ir, im_vis, im_seg_float])

    def fullseq(self, withcal=True):
        """ generator yields full sequence
        {ims, ims_ori, xyzs, Ks}"""
        allframesleft, allframesright = self.extractallframes()
        #print(len(allframes))
        for frameleft, frameright in zip(allframesleft, allframesright):
            assert frameleft.shape == (512, 640, 3)
            assert frameright.shape == (512, 640, 3)
            image = self.transform(frameleft)
            image_right = self.transform(frameright)
            im_ori = to_ori(image)
            im_ori_right = to_ori(image_right)
            ims = [image]
            ims_right = [image_right]
            ims_ori = [im_ori]
            ims_ori_right = [im_ori_right]
            xyzs = [image.new_zeros((1))]
            if withcal:
                K = torch.tensor([self.K])
                Q = torch.tensor([self.Q])
                disparitypad = torch.tensor([np.float32(self.disparitypad)])

                out = {
                    "ims": ims,
                    "ims_right": ims_right,
                    "ims_ori": ims_ori,
                    "ims_ori_right": ims_ori_right,
                    "xyzs": xyzs,
                    "Ks": K,
                    "Qs": Q,
                    "disparitypads": disparitypad
                }
            else:
                out = {
                    "ims": ims,
                    "ims_right": ims_right,
                    "ims_ori": ims_ori,
                    "ims_ori_right": ims_ori_right,
                    "xyzs": xyzs,
                }
            yield out

    def extractallframes(self):
        """ extracts whole sequence into tmpdir, and then loads the frames. Overcomplicated"""
        def getframes(filename):
            size = (640, 512)
            usetmpdir = False
            if usetmpdir:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    self.extractfullvideo(filename, tmpdirname, 'visible')
                    framenames = sorted(os.listdir(tmpdirname))
                    vidframes = []
                    for frame in framenames:
                        frame = cv2.resize(loadimcv(os.path.join(str(tmpdirname),frame)), (640, 512))
                        #cv2.imshow('test_cv', frame)
                        #cv2.waitKey(1)
                        vidframes.append(frame)
                    return vidframes
            else:
                frames = self.extractfullvideopipe(filename, 'visible')
                return [cv2.resize(x, size) for x in frames]
        leftvidframes = getframes(self.leftvidname)
        rightvidframes = getframes(self.rightvidname)
        if "SKIP" in os.environ:
            SKIP = int(os.environ['SKIP'])
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
        """ extracts video to outdir/frames, suffix with segname"""
        outstr = f'{str(outdir)}/%06d_{segname}.png'
        command = [ 'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', str(videoname),
        outstr]
        process = sp.run(command)

    @staticmethod
    def extractfullvideopipe(videoname, segname):
        """ extracts video to list, suffix with segname RGB format"""
        command = [ 'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', str(videoname),
        '-pix_fmt', 'rgb24',
        '-f', 'rawvideo', '-']
        pipe = sp.Popen(command, stdout=sp.PIPE)
        cnt = 0
        W = 1280
        H = 1024
        frames = []
        while True:
            cnt += 1
            raw_image = pipe.stdout.read(W*H*3)
            image = np.fromstring(raw_image, dtype='uint8')
            if image.shape[0] == 0:
                break
            else:
                image = image.reshape((H, W, 3))
                frames.append(image)
            #cv2.imshow('test', image)
            #cv2.waitKey(1)
        pipe.stdout.close()
        pipe.wait() 
        return frames

    @staticmethod
    def extractfirstframefromname(videoname, outdir, segname):
        """ extracts video to outdir/frames, suffix with segname"""
        outstr = f'{str(outdir)}/%06d_{segname}.png'
        command = [ 'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', str(videoname),
        '-vframes', '1',
        outstr]
        process = sp.run(command)

    def extractfirstframe(self):
        """ extracts first frame into tmpdir, and then loads said frame. Overcomplicated"""
        def getframes(filename):
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.extractfirstframefromname(filename, tmpdirname, 'visible')
                framenames = sorted(os.listdir(tmpdirname))
                vidframes = []
                for frame in framenames:
                    frame = cv2.resize(loadimcv(os.path.join(str(tmpdirname),frame)), (640, 512))
                    vidframes.append(frame)
                return vidframes
        leftvidframe = getframes(self.leftvidname)[0]
        return leftvidframe
