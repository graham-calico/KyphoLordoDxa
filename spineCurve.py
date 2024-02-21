import os, cv2, sys
import tensorflow as tf, keras, keras_segmentation
import argparse, os, numpy as np, math
import scipy.odr

perms = {}
perms["spine_mask_model"] = 'models/msk.43'
perms["spine_mask_model_type"] = "vgg_unet"
perms["spine_mask_n_classes"] = "2"
perms["spine_mask_input_height"] = "384"
perms["spine_mask_input_width"] = "192"
perms["spine_mask_num_to_name"] = {1:'body'}
  
perms["bridge_mod"] = "models/brBox.frInfGr.pb"
perms["bridge_label"] = "models/brBox.label.pbtxt"
perms["bridge_num"] = "14"
perms["bridge_min_score"] = "0"

perms["sideface_mod"] = "models/side.pb"
perms["sideface_label"] = "models/side.labels.txt"

defaults = {}
defaults["kypho"] = {}
defaults["lordo"] = {}
#
defaults["kypho"]["box_fraction"] = 1.0 / 3.0
defaults["kypho"]["bottom_cut"] = 0.0
defaults["kypho"]["thorax_end"] = 11.5
#
defaults["lordo"]["box_fraction"] = 1.0 / 3.0
defaults["lordo"]["bottom_cut"] = 0.06
defaults["lordo"]["thorax_end"] = 10.25


def main():
  # start the app
  ap = argparse.ArgumentParser()
  ap.add_argument("-i","--input_dir",
                  help="input directory of images to be scored (or .txt file listing images)")
  ap.add_argument("-o","--output_file",
                  help="output file of box locations (or no entry for stdout)",
                  default='stdout')
  ap.add_argument("-d","--drawing_dir",
                  help="optional output directory for marked-up images")
  ap.add_argument("-a","--annot_file",
                  help="a file of annotated images for performance comparison")
  # which angle? default: thoracic
  ap.add_argument("--lumbar",
                  help="measure lumbar kyphosis/lordosis instead of thoracic (thor=default)",
                  action="store_true")
  # data augmentation
  ap.add_argument("--aug_flip",
                  help="score each image twice, with a horizontal flip",
                  action='store_true')
  ap.add_argument("--aug_tilt",
                  help="addionally score each image tilted in each dir by this # DEGREES",
                  action='append')
  # forced sided-ness
  ap.add_argument("--side_facing",
                  help="otional: 'left' or 'right' if people in images are known to face that way",
                  default='none')
  # optional parameter adjustment
  ap.add_argument("--box_fraction",
                  help="otional: adjust the length of the asymptote; (0,1) interval, recommend < 0.5",
                  default='default')
  ap.add_argument("--thoracic_bottom",
                  help="otional: which bridge defines thoracic spine bottom (1-14, 1-indexed from top)",
                  default='default')
  ap.add_argument("--bottom_cut",
                  help="otional: cut some fraction off the bottom of the box; [0,1) interval, recommend < 0.2",
                  default='default')
  # legacy mode (original rather than corrected algorithm)
  ap.add_argument("--legacy",
                  help="sets the algorithm to its original algorithm, prior to update (see docs)",
                  action="store_true")
  args = vars(ap.parse_args())

  if args['side_facing']=='none':
    knownSideFace = False
    sideFaceMod = TfClassifier(perms["sideface_mod"],perms["sideface_label"])
  elif args['side_facing']=='right' or args['side_facing']=='left':
    knownSideFace = True
    sideFace = args['side_facing']
    sideFaceMod = None
  else: raise ValueError("--side_facing options: 'right' or 'left' (or don't call)")
  
  spineMaskMod = KerasMasker(perms["spine_mask_model_type"],
                             int(perms["spine_mask_n_classes"]),
                             int(perms["spine_mask_input_width"]),
                             int(perms["spine_mask_input_height"]),
                             perms["spine_mask_model"],
                             numToName=perms["spine_mask_num_to_name"])
  bridgeMod = TfObjectIdentifier(perms["bridge_mod"],perms["bridge_label"],
                                 int(perms["bridge_num"]))

  scorer = CurveScorer(spineMaskMod,bridgeMod,sideFaceMod)
  if knownSideFace: scorer.setDirection(sideFace)
  if args["aug_flip"]:
    scorer.addAugmentHorFlip()
  if args["aug_tilt"]:
    for tiltArg in args["aug_tilt"]:
      tiltDeg = float(tiltArg)
      if tiltDeg <= 0: raise ValueError('tilts must be POSITIVE degrees')
      if tiltDeg >= 45: raise ValueError('tilts must be under 45 degrees')
      scorer.addAugmentTilt(tiltDeg)
  if args["legacy"]:
    print("USING LEGACY ALGORITHM")
    scorer.setLegacyAlgorithm()
  curveMode = 'kypho'
  if args["lumbar"]:
    curveMode = 'lordo'
    scorer.setLumbarMode()
  # default is different depending on mode
  if args["box_fraction"]=='default':
    scorer.setBoxFraction(defaults[curveMode]["box_fraction"])
  else:
    scorer.setBoxFraction(float(args["box_fraction"]))
  # default is different depending on mode
  if args["bottom_cut"]=='default':
    scorer.setBottomCut(defaults[curveMode]["bottom_cut"])
  else:
    scorer.setBottomCut(float(args["bottom_cut"]))
  # default is different depending on mode    
  if args["thoracic_bottom"]=='default':
    scorer.setThoraxEnd(defaults[curveMode]["thorax_end"])
  else:
    tEndN = float(args["thoracic_bottom"])
    scorer.setThoraxEnd( tEndN )

  if args["input_dir"]:
    if os.path.isdir(args["input_dir"]):
      imgLister = ImageDirLister(args["input_dir"])
    elif os.path.isfile(args["input_dir"]):
      imgLister = ImageFileLister(args["input_dir"])
    else:
      sys.stderr.write(args["input_dir"])
      raise ValueError("input is nether a directory nor a file")
    imgMang = ImageDirScorer(scorer,imgLister)
    if args["drawing_dir"]: imgMang.setDrawingDir(args["drawing_dir"])
    if args["output_file"]: outfName = args["output_file"]
    else: outfName = 'stdout'
    imgMang.scoreImages(outfName)
    
  if args["annot_file"]:
    perfMang = PerformanceAnalyzer(args["annot_file"])
    perfMang.scoreImages(scorer)
    
  print('done.')


# writes the output boxes
class CurveScorer:
  def __init__(self, maskMod, bridgeBoxMod, sideMod):
    self._maskMod = maskMod
    self._bridgeBoxMod = bridgeBoxMod
    self._sideMod = sideMod
    # verify that mask value 1 is the body of the vertebra
    if maskMod.getName(1)!='body':
      raise ValueError('is this the correct model???')
    self._spineVal = 1
    self._nBins = 90
    # verify sideMod value labels
    if sideMod!=None:
      sfLabelL = sideMod.labels()
      if len(sfLabelL)!=2:
        raise ValueError('right/left model wrong # classes')
      # these will raise exceptions if not found
      z = sfLabelL.index('left')
      z = sfLabelL.index('right')
    # augmentations
    self._aug_spineFlip = False
    self._nrTiltRads = {}
    # how I'm choosing to down-weight angle-based
    # augmentations, which will presumably be of
    # lower quality as the angle increases
    self._calcRadWgt = lambda r: 0.1**(4*r)
    # for dealing with sidedness
    self._sideSpec = None
    # for dealing with way-too-big images
    self._maxVertDim = 800
    # thoracic or lumbar?
    self._lumbarInstead = False
    # VALUES THAT MUST BE SET:
    # how much of the box edge to use
    # for calculating slopes
    self._isSet_boxFr = False
    # extra bottom-of-box trimming
    self._isSet_botCut = False
    # border between thorax & lumbar
    self._isSet_thorEnd = False
    # ALLOWING FOR MODIFICATION made on 9/1/23:
    # the original ("legacy") version used for
    # the analysis did not actually refine the
    # trace after adjusting the axis around the
    # region of interest.  I'd always intended
    # for it to do this, and updated it to do it.
    # Differences in the output are negligable,
    # but nonetheless, this variable allows the
    # original output to be generated
    self._legacyAlgorithm = False

  # takes & returns a tuple: (start,end), 1-indexed
  def getThoraxEnd(self):
    if not(self._isSet_thorEnd):
      raise ValueError('thoracic bottom requested but not set')
    return self._thoraxEnd
  def setThoraxEnd(self,end):
    if end < 0:
      sys.stderr.write('specified end:\t'+str(end)+'\n')
      raise ValueError("thorax end must be >=1 (1-indexed)")
    if end > self._bridgeBoxMod.nBoxes():
      sys.stderr.write('specified end:\t'+str(end)+'\n')
      sys.stderr.write('maximum end:\t'+str(self._bridgeBoxMod.nBoxes())+'\n')
      raise ValueError("thorax end must be <=MAX")
    self._isSet_thorEnd = True
    self._thoraxEnd = float(end)
  
  # how far from the edge of the thorax region to
  # extend, as a fraction of the thorx region's length
  def getBoxFraction(self):
    if not(self._isSet_boxFr):
      raise ValueError('box fraction requested but not set')
    return self._boxFraction
  def setBoxFraction(self,boxF):
    if boxF <= 0:
      raise ValueError("box fraction must be positive, non-zero")
    if boxF >= 1:
      raise ValueError("box fraction must be less than one")
    self._isSet_boxFr = True
    self._boxFraction = boxF
  def getBottomCut(self):
    if not(self._isSet_botCut):
      raise ValueError('bottom cut requested but not set')
    return self._bottomCut
  def setBottomCut(self,botCut):
    if botCut < 0:
      raise ValueError("bottom cut must be zero or positive")
    if botCut >= 1:
      raise ValueError("bottom cut must be less than one")
    self._isSet_botCut = True
    self._bottomCut = botCut
  def setDirection(self,side):
    if side=='none' or side==None: self._sideSpec = None
    elif side=='right': self._sideSpec = side
    elif side=='left': self._sideSpec = side
    else: raise ValueError('side specification must be left, right, or none')
  def setLumbarMode(self):
    self._lumbarInstead = True
  def setLegacyAlgorithm(self):
    self._legacyAlgorithm = True
  def addAugmentHorFlip(self): self._aug_spineFlip = True
  def addAugmentTilt(self,tiltDeg):
    if tiltDeg < 0: tiltDeg *= -1
    if tiltDeg >= 45: raise ValueError("tilt must be under 45 degrees")
    if tiltDeg!=0:
      tiltRad = math.radians(tiltDeg)
      self._nrTiltRads[tiltRad] = None
  
  # alternative method to produce marked-up image
  def markImg(self,img):
    return self.scoreImg(img,doMarking=True)
  def scoreImg(self,img,doMarking=False):
    if not(self._isSet_boxFr): raise ValueError('box fraction not set')
    if not(self._isSet_botCut): raise ValueError('bottom cut not set')
    # down-size the image if it is excessively large
    img = self._helpResize(img)
    # set up a mark-up image, if appropriate
    if doMarking: mImg = np.copy(img)
    else: mImg = None
    # lists of values for cobb angles, left/right choice, weights
    scoreL,pRightL,weightL = [],[],[]
    # initiate lists with values for first image;
    # only allow marking-up here
    scI,rtP = self._oneScoreHelper(img,False,doMarking,mImg)
    scoreL.append(scI)
    pRightL.append(rtP)
    weightL.append(1.0)
    # OPTION: mark twice (kypho + lordo)
    if doMarking and False:
      print("DOUBLE MARKING")
      # reference values (Alt for this image only)
      lumbarReal = self._lumbarInstead
      lumbarAlt = not(self._lumbarInstead)
      # flip kypho/lordo mode, mark image, then flip back
      self._lumbarInstead = lumbarAlt
      trashA,trashB = self._oneScoreHelper(img,False,doMarking,mImg)
      self._lumbarInstead = lumbarReal
    # AUG: flip of raw image (equal weight)
    if self._aug_spineFlip:
      flipImg = np.flip(np.copy(img),1)
      scI,rtP = self._oneScoreHelper(flipImg,True,False,None)
      scoreL.append(scI)
      pRightL.append(rtP)
      weightL.append(1.0)
    # AUG: tilt by spec amounts in each direction
    for tRad in self._nrTiltRads.keys():
      for tR in [tRad,-tRad]:
        tiltImg = self._helpTiltImg(img,tR)
        scI,rtP = self._oneScoreHelper(tiltImg,False,False,None)
        scoreL.append(scI)
        pRightL.append(rtP)
        weightL.append(self._calcRadWgt(tR))
        # AUG: also apply flip to tilted image
        if self._aug_spineFlip:
          flipTiltImg = np.flip(np.copy(tiltImg),1)
          scI,rtP = self._oneScoreHelper(flipTiltImg,True,False,None)
          scoreL.append(scI)
          pRightL.append(rtP)
          weightL.append(self._calcRadWgt(tR))
    # calculate the average cobb est acorss augmentations
    cobbV = np.average(scoreL,weights=weightL)
    # flip the angle sign if the image is overall more likely
    # to be left-hand-facing than right-hand-facing
    rightP = np.average(pRightL,weights=weightL)
    if rightP < 0.5: cobbV *= -1
    if doMarking: return cobbV, mImg
    else: return cobbV

  def _helpTiltImg(self,img,rotAng):
    h,w = img.shape[:2]
    wOldAng = np.arctan(float(-h)/w)
    wNewAng = wOldAng + abs(rotAng)
    wF = math.cos(wNewAng)/math.cos(wOldAng)
    hOldAng = np.arctan(float(h)/w)
    hNewAng = hOldAng + abs(rotAng)
    hF = math.sin(hNewAng)/math.sin(hOldAng)
    dimF = max(wF,hF)
    wDif = int( (w - (w/dimF)) * 0.5)
    hDif = int( (h - (h/dimF)) * 0.5)
    M = cv2.getRotationMatrix2D((w/2, h/2), math.degrees(rotAng), 1.0)
    rotImg = cv2.warpAffine(img, M, (w,h))
    return rotImg[hDif:h-hDif,wDif:w-wDif,:]
    
  def _helpResize(self,img):
    imgH,imgW = img.shape[:2]
    if imgH <= self._maxVertDim: return img
    else:
      scaleFct = float(self._maxVertDim)/imgH
      newH = int(imgH*scaleFct)
      newW = int(imgW*scaleFct)
      if newW < 1: newW = 1
      return cv2.resize(img,(newW,newH))

  # RETURNS: angle assuming right-facing input image,
  #          prob(original input image was rt-facing)
  #   both of the above: use 'flipped' to infer relationship
  #   between the input image and the one being scored
  def _oneScoreHelper(self,img,flipped,doMarks,markImg):
    scI = self.internalScoreImg(img,doMarks=doMarks,markImg=markImg)
    if self._sideSpec==None: clRes = self._sideMod.getClasses(img)
    if flipped:
      scI *= -1
      if self._sideSpec=='right': rtP = 0.0
      elif self._sideSpec=='left': rtP = 1.0
      else: rtP = clRes.score('left')
    else:
      if self._sideSpec=='right': rtP = 1.0
      elif self._sideSpec=='left': rtP = 0.0
      else: rtP = clRes.score('right')
    return scI,rtP

  def _ptListToTwoArrays(self,xyL):
    aX = np.array(list(map(lambda xy: xy[0], xyL)))
    aY = np.array(list(map(lambda xy: xy[1], xyL)))
    return aX,aY
  def _filtXyByY(self,xyL,minY,maxY):
    fF = lambda xy: xy[1] >= minY and xy[1] < maxY
    return list(filter(fF, xyL))
  
  def _getBridgeBoxL(self,img):
    """calls the obj-detect model and gets the most-
       confidently-identified boxes, using the model-
       defined max number of boxes.
    """
    boxL = self._bridgeBoxMod.getBoxes(img)
    #boxL = list(filter(lambda b: b.score() >= self._bridgeBoxMod.minScr(), boxL))
    if len(boxL) < self._bridgeBoxMod.nBoxes(): return []
    if len(boxL) > self._bridgeBoxMod.nBoxes():
      # the n is the tiebreaker
      boxL = [(boxL[n].score(),n,boxL[n]) for n in range(len(boxL))]
      boxL.sort()
      boxL.reverse()
      boxL = boxL[:self._bridgeBoxMod.nBoxes()]
      boxL = [b for (s,n,b) in boxL]
    return boxL
  
  def _getThoracicBox(self,img):
    # get the bridge boxes & sort on y-axis
    boxL = self._getBridgeBoxL(img)
    if len(boxL)==0: return Box(0,0,0,0,0.0,'no boxes')
    # # the box begins at the top of the highest box (min y value)
    # yPosBeg = min(list(map(lambda b: b.yMin(), boxL)))
    # Changing this to just be the top of the image
    yPosBeg = 0.0
    # the end of the thoracic spine defined by counting down the vertebrae
    yPosL = list(map(lambda b: np.mean([b.yMin(),b.yMax()]), boxL))
    yPosL.sort()
    tEnd,itEnd = self._thoraxEnd,int(self._thoraxEnd)
    # calculate the position between y values
    wtMean = lambda a,b,w: a*w + b*(1.0 - w)
    if itEnd==len(yPosL): yPosEnd = yPosL[-1]
    else: yPosEnd = wtMean(yPosL[itEnd-1],yPosL[itEnd-2],tEnd-itEnd)
    # change things up if looking at the lumbar region
    if self._lumbarInstead:
      # edge of the thoracic region is now the "top"
      yPosBeg = yPosEnd
      # bottom of the image is the bottom of the box too
      yPosEnd = float(img.shape[0])
    return Box(0,yPosBeg,0,yPosEnd,1.0,'found box')

  # NOTE: there are escapes to return values of >360
  # (exact value tells me where failure happened)
  # if the model fails (no mask, no box volume, etc)
  def internalScoreImg(self,img,doMarks=False,markImg=None):
    mask = self._maskMod.getMask(img)
    # NOTE: in lumbar mode, the "thoracic box" will actually
    # be the "lumbar box"
    thorBox = self._getThoracicBox(img)
    if thorBox.yMin()==thorBox.yMax(): return 361.0
    # get the trace from the overall mask
    maskX,maskY,mArea = self.getMaskAsPoints(mask)
    if mArea==0: return 362.0
    maskOdr = self.getOdrLine(maskX,maskY)
    mTraceXyL = self.getTraceFromPoints((maskX,maskY),mArea,maskOdr)
    ###
    ### LEGACY VERSION (pre-9/1/23)
    ###
    if self._legacyAlgorithm:
      # refine the trace using the upper-spine trace
      thorXyL = self._filtXyByY(mTraceXyL,thorBox.yMin(),thorBox.yMax())
      if len(thorXyL)==0: return 363.0
      thorX,thorY = self._ptListToTwoArrays(thorXyL)
      thorOdr = self.getOdrLine(thorX,thorY)
      tTraceXyL = self.getTraceFromPoints((thorX,thorY),len(thorXyL),thorOdr)
      # trim the edges of the region (possibly)
      thorBox = self._refineBoxUsingMask((maskX,maskY),mArea,maskOdr,thorBox)
    ###
    ### UPDATED VERSION (post-9/1/23)
    ### swapping order of box refinement,
    ### modifying input to generate tTraceXyL
    ###
    else:
      # trim the edges of the region (possibly)
      thorBox = self._refineBoxUsingMask((maskX,maskY),mArea,maskOdr,thorBox)
      # define a new, region-specific ODR
      thorXyL = self._filtXyByY(mTraceXyL,thorBox.yMin(),thorBox.yMax())
      if len(thorXyL)==0: return 363.0
      thorX,thorY = self._ptListToTwoArrays(thorXyL)
      thorOdr = self.getOdrLine(thorX,thorY)
      # refine the trace using the region-specific ODR
      tTraceXyL = self.getTraceFromPoints((maskX,maskY),mArea,thorOdr)
    # get ODRs for top/bottom of thoracic trace
    upY = thorBox.yMin()
    lowY = thorBox.yMax()
    upMidY,lowMidY = self.getEstimateRegions((maskX,maskY),mArea,maskOdr,thorBox)
    upThorXyL = self._filtXyByY(tTraceXyL,upY,upMidY)
    if len(upThorXyL)==0: return 364.0
    lowThorXyL = self._filtXyByY(tTraceXyL,lowMidY,lowY)
    if len(lowThorXyL)==0: return 365.0
    upTX,upTY = self._ptListToTwoArrays(upThorXyL)
    lowTX,lowTY = self._ptListToTwoArrays(lowThorXyL)
    upThorOdr = self.getOdrLine(upTX,upTY)
    lowThorOdr = self.getOdrLine(lowTX,lowTY)
    # calculate the angle difference
    upAng = upThorOdr.angle()
    lowAng = lowThorOdr.angle()
    cobbAng = math.degrees(lowAng - upAng)
    while abs(cobbAng) >= 360:
      if cobbAng > 0: cobbAng -= 360
      else: cobbAng += 360
    if abs(cobbAng) > 90: cobbAng += 180
    if cobbAng >= 180: cobbAng -= 360
    # mark up the image
    if doMarks:
      # initial region-of-interest borders
      if False:
        xLineMax = markImg.shape[1]-1
        barYa,barYb = int(thorBox.yMin()),int(thorBox.yMax())
        cv2.line(markImg,(0,barYa),(xLineMax,barYa),(100,255,200),3)
        cv2.line(markImg,(0,barYb),(xLineMax,barYb),(100,255,200),3)
      # re-color mask
      if False:
        bmask = np.where(mask==self._spineVal,1,0)
        markImg[:,:,0] = bmask * 255
      # mark all trace points & axis
      if False:
        self.markPoints(mTraceXyL,markImg,(100,0,255),3)
        self.markAxis(markImg,maskOdr,(100,255,255),
                      (0,markImg.shape[0]-1),2)
      # mark the refined axis (and original axis)
      if False:
        if self._lumbarInstead:
          lineColor = (55,255,20)
          pointColor = (100,255,0)
          dotSize = 2
        else:
          lineColor = (255,75,255)
          pointColor = (175,50,225)
          dotSize = 4
        self.markPoints(tTraceXyL,markImg,pointColor,dotSize)
        self.markAxis(markImg,thorOdr,lineColor,
                      (0,markImg.shape[0]-1),2)
      # mark points for upper & lower
      if True:
        self.markPoints(upThorXyL,markImg,(0,100,255),5)
        self.markPoints(lowThorXyL,markImg,(100,0,255),5)
      # draw the two lines
      if True:
        self.markAxis(markImg,upThorOdr,(100,255,255),(upY,upMidY))
        self.markAxis(markImg,lowThorOdr,(100,255,255),(lowY,lowMidY))
    # change the sign of the angle for lumbar
    if self._lumbarInstead: cobbAng *= -1.0
    return cobbAng
  
  def getOdrLine(self,mX,mY):
    # calculate the ODR regression line
    odrBeta = self._getOdrBeta(mX,mY)
    odrCx,odrCy = self._getCenter(mX,mY)
    odrLine = self._Line( (odrCx,odrCy),
                          (odrCx+10, odrCy+(odrBeta*10)) )
    return odrLine
  def getMaskAsPoints(self,mask):
    # new
    indexA = np.indices( mask.shape )
    indexAx,indexAy = indexA[1,:,:],indexA[0,:,:]
    bmask = np.where(mask==self._spineVal)
    mX = indexAx[bmask]
    mY = indexAy[bmask]
    mArea = mY.shape[0]
    return mX,mY,mArea

  def markPoints(self,xyL,markImg,color,ptSize,trLines=False):
    for xP,yP in xyL:
      # defensive line against legacy code error that
      # produced NaN trace points:
      if not(np.isnan(xP)) and not (np.isnan(yP)):
        cv2.circle(markImg,(int(xP),int(yP)),ptSize,color,-1)
    if trLines:
      for n in range(1,len(xyL)):
        xyA,xyB = tuple(map(int,xyL[n-1])),tuple(map(int,xyL[n]))
        cv2.line(markImg,xyA,xyB,color,2)

  #(100,255,255)
  def markAxis(self,markImg,odrLine,color,yMinMax,lineThick=3):
    minY,maxY = yMinMax
    minX,maxX = odrLine.getX(minY),odrLine.getX(maxY)
    TEMP_axisLen = np.sqrt( (minX-maxX)**2 + (minY-maxY)**2 )
    cv2.line(markImg,(int(minX),int(minY)),
             (int(maxX),int(maxY)),color,lineThick)

  # trim the bottom of the box
  def _refineBoxUsingMask(self,maskXY,mArea,odrLine,thorBox):
    if self._bottomCut <= 0: return thorBox
    else:
      edgeUp = thorBox.yMin()
      mX,mY = maskXY
      # used to make the line here
      pintX,pintY = odrLine.perpIntersect( (mX,mY) )
      # get indexes using y position on spine axis
      pIndYL = [pintY[n] for n in range(mArea)]
      # filter just those within the thoracic box
      isInBox = lambda y: y >= thorBox.yMin() and y <= thorBox.yMax()
      pIndYL = np.array(list(filter(isInBox, pIndYL)))
      # sort indexes using position on spine axis, then
      # use percentiles to get the thresholds that cover
      # appropriate fractions of the density
      edgeLow = np.percentile(pIndYL,(1-self._bottomCut)*100)
      return Box(0,edgeUp,0,edgeLow,1.0,'refined box')

  # maskXY is a tuple of nuber arrays for the x and
  #        y coordinates of each masked pixel
  def getEstimateRegions(self,maskXY,mArea,odrLine,thorBox):
    mX,mY = maskXY
    # used to make the line here
    pintX,pintY = odrLine.perpIntersect( (mX,mY) )
    # get indexes using y position on spine axis
    pIndYL = [pintY[n] for n in range(mArea)]
    # filter just those within the thoracic box
    isInBox = lambda y: y >= thorBox.yMin() and y <= thorBox.yMax()
    pIndYL = np.array(list(filter(isInBox, pIndYL)))
    # sort indexes using position on spine axis, then
    # use percentiles to get the thresholds that cover
    # appropriate fractions of the density
    edgePctlA = np.array([self._boxFraction,1-self._boxFraction]) * 100
    edgeUp,edgeLow = np.percentile(pIndYL,edgePctlA)
    return edgeUp,edgeLow

  # maskXY is a tuple of nuber arrays for the x and
  #        y coordinates of each masked pixel
  def getTraceFromPoints(self,maskXY,mArea,odrLine):
    mX,mY = maskXY
    # used to make the line here
    pintX,pintY = odrLine.perpIntersect( (mX,mY) )
    # sort indexes using position on spine axis
    pIndNL = [(pintY[n],n) for n in range(mArea)]
    pIndNL.sort()
    # organize the points into bins along the axis;
    # this list is point indexes in the original arrays
    pnBinL = []
    for n in range(self._nBins):
      targN = float(mArea * (n+1)) / self._nBins
      pnBinL.append( [] )
      n2 = sum(list(map(len,pnBinL)))
      # in case there's any rounding error at end-of-list
      while n2 < targN and n2 < len(pIndNL):
        pnBinL[-1].append(pIndNL[n2][1])
        n2 += 1
    # for each bin, get the average position
    avgXyL = []
    for pnL in pnBinL:
      binXL = list(map(lambda n: mX[n], pnL))
      binYL = list(map(lambda n: mY[n], pnL))
      avgX,avgY = np.mean(binXL),np.mean(binYL)
      avgXyL.append( (avgX,avgY) )
    return avgXyL
  # generated based on the scipy.ord doc page:
  # https://docs.scipy.org/doc/scipy/reference/odr.html
  # x,y are 1-D arrays
  def _getOdrBeta(self,x,y):
    beta0 = np.polyfit(x,y,1)
    # the func to fit against
    f = lambda B,x: B[0]*x + B[1]
    linear = scipy.odr.Model(f)
    data = scipy.odr.RealData(x,y)
    odr = scipy.odr.ODR(data, linear, beta0=beta0)
    return odr.run().beta[0]
  def _getCenter(self,x,y):
    return np.mean(x),np.mean(y)
  class _Line:
    def __init__(self,xy1,xy2):
      if xy1==xy2: raise ValueError('points cant be equal for line')
      self._xy1 = xy1
      self._xy2 = xy2
      self._run = xy2[0]-xy1[0]
      self._rise = xy2[1]-xy1[1]
      self._dist = np.sqrt(self._run**2 + self._rise**2)
    def inputSegLen(self): return self._dist
    def angle(self):
      if self._dist == 0: return 0.0
      # negative since low-Y means "up" in images
      return np.arctan2(-self._rise,self._run)
    def dist(self,xy):
      x0,y0 = xy
      x1,y1 = self._xy1
      x2,y2 = self._xy2
      # eq for dist from line defined by 2 points
      numer = np.abs( (x2-x1)*(y1-y0) - (x1-x0)*(y2-y1) )
      denom = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
      return float(numer)/denom
    def getY(self,x):
      if self._rise == 0: return x*0 + self._xy1[1]
      elif self._run == 0: return x + float("nan")
      else:
        m = self._rise / self._run
        x1,y1 = self._xy1
        return y1 + m*x - m*x1
    def getX(self,y):
      if self._rise == 0: return y + float("nan")
      elif self._run == 0: return y*0 + self._xy1[0]
      else:
        m = self._rise / self._run
        x1,y1 = self._xy1
        return (y + m*x1 - y1) / m
    def perpIntersect(self,xy):
      dataX,dataY = xy
      if self._run==0: return self._xy1[0],dataY
      else:
        axSlope = self._rise / self._run
        axX,axY = self._xy1
        newY = axY + (axSlope**2 * dataY) - axSlope*(axX - dataX)
        newY /= axSlope**2 + 1
        newX = (axSlope*axX + newY - axY) / axSlope
        return newX,newY
    def perpMoveFromXY(self,xy,dist,direct):
      x,y = xy
      # direct is backwards from a normal x,y plane
      # since in images, higher y values are lower
      # in the image
      if direct == "clock":
        run = -self._rise
        rise = self._run
      elif direct == "counter":
        run = self._rise
        rise = -self._run
      else:
        raise ValueError('direct must be "clock" or "counter"')
      # adjust rise & run by this scale factor
      sc = dist / self._dist
      return x + sc*run, y + sc*rise
    def twoLineIntersect(self,xy3,xy4):
      if xy3==xy4: raise ValueError('new line requires two different points')
      # variables defined as described in:
      # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
      x1,y1 = self._xy1
      x2,y2 = self._xy2
      x3,y3 = xy3
      x4,y4 = xy4
      D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
      Px_numer = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)
      Py_numer = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)
      return Px_numer/D, Py_numer/D



class ImageDirLister:
  def __init__(self,hostDir,append='.png'):
    # check that the host dir exists
    if not(os.path.isdir(hostDir)):
      raise ValueError("host dir doesn't exist")
    self._hostD = os.path.abspath(hostDir)
    self._append = append
  def getImgFiles(self):
    imgFL = os.listdir(self._hostD)
    imgFL.sort()
    aLen = len(self._append)
    imgFL = list(filter(lambda i: i[-aLen:]==self._append, imgFL))
    imgFL = list(map(lambda i: os.path.join(self._hostD,i), imgFL))
    return imgFL
class ImageFileLister:
  def __init__(self,fileOfFiles):
    # check that the host dir exists
    if not(os.path.isfile(fileOfFiles)):
      raise ValueError("file-of-files doesn't exist")
    self._fofName = fileOfFiles
  def getImgFiles(self):
    f = open(self._fofName)
    imgFL = f.readlines()
    f.close()
    imgFL = list(map(lambda i: i.rstrip(), imgFL))
    return imgFL
  
class ImageDirScorer:
  def __init__(self,scorer,fileLister):
    self._scorer = scorer
    self._fileLister = fileLister
    self._draw = False
  def setDrawingDir(self,dirName):
    self._draw = True
    self._drawDir = dirName
  def scoreImages(self,outfileName):
    imgFL = self._fileLister.getImgFiles()
    print("Analyzing "+str(len(imgFL))+" images.")
    if outfileName=='stdout':
      outf = sys.stdout
      progress = NullDotWriter()
    else:
      outf = open(outfileName,'w')
      progress = DotWriter(5,50,250)
    count = 0
    for imgF in imgFL:
      progress.tick()
      if len(imgF.split('.')) < 2: aLen = 0
      else: aLen = len(imgF.split('.')[-1]) + 1
      imgName = os.path.basename(imgF)[:-aLen]
      outf.write(imgName+'\t')
      outf.flush()
      img = cv2.imread(imgF)
      if self._draw:
        score,outImg = self._scorer.markImg(img)
        outImgName = os.path.join(self._drawDir,os.path.basename(imgF)) + '.jpg'
        if os.path.isfile(outImgName): raise ValueError('marked file exists')
        cv2.imwrite(outImgName,outImg)
      else:
        score = self._scorer.scoreImg(img)
      outf.write(str(score)+'\n')
      outf.flush()
    if outf!=sys.stdout: outf.close()

class DotWriter:
  def __init__(self,perDot,perBar,perLine):
    self._pDot = perDot
    self._pBar = perBar
    self._pLine = perLine
    self._count = 0
  def tick(self):
    self._count += 1
    if self._count % self._pBar == 0: sys.stdout.write('|')
    elif self._count % self._pDot == 0: sys.stdout.write('.')
    if self._count % self._pLine == 0: sys.stdout.write('\n')
    sys.stdout.flush()
class NullDotWriter:
  def __init__(self): pass
  def tick(self): pass

class PerformanceAnalyzer:
  def __init__(self,annotFile):
    self._imgfToScore = {}
    f = open(annotFile)
    line = f.readline()
    while line:
      if line[0]!=">":
        imgF = line.strip()
        self._imgfToScore[imgF] = 0.0
      else:
        cols = line[1:].rstrip().split('\t')
        # the categories will be in the last column, either "brN" or "Br_N"
        # where N is 0, 1, 2, or 3;
        # "FV_" provides the option to give a float value
        if cols[-1].find('FV_')==0:
          scr = float(cols[-1].split('_')[1])
        else: scr = int(cols[-1][-1])
        self._imgfToScore[imgF] += scr
      line = f.readline()
    f.close()
  def scoreImages(self,scorer):
    print("Analyzing "+str(len(self._imgfToScore))+" images.")    
    annotL,modelL = [],[]
    progress = DotWriter(5,50,250)
    for imgF in self._imgfToScore.keys():
      progress.tick()
      annotL.append(self._imgfToScore[imgF])
      img = cv2.imread(imgF)
      score = scorer.scoreImg(img)
      modelL.append(score)
    sys.stdout.write('\n')
    print(str(scipy.stats.linregress(annotL,modelL)))
      
# separate out the mask-drawing
class KerasMasker:
  def __init__(self,segMod,nClass,inW,inH,modFile,numToName=None):
    self._segMod,self._nC = segMod,nClass
    self._inW,self._inH,self._modFile = inW,inH,modFile
    modType = keras_segmentation.models.model_from_name[segMod]
    self._model = modType(n_classes=nClass,input_height=inH,input_width=inW)
    self._model.load_weights(modFile)
    self._inWidHgt = (inW,inH)
    self._outShape = (inH,inW,nClass)
    self._numToName = {}
    if numToName: self._numToName.update(numToName)
    else:
      for n in range(nClass): self._numToName[n+1] = str(n+1)
  def getName(self,chNum): return self._numToName[chNum]
  def getMask(self,image):
    h,w = image.shape[:2]
    seg = self._model.predict_segmentation(image)
    # now I need to re-size the masks to match the image
    hS,wS = seg.shape
    segI = np.zeros( (hS,wS,3) )
    segI[:,:,0] = seg
    gt = keras_segmentation.data_utils.data_loader.get_segmentation_arr(segI,self._nC,w,h)
    gt = gt.argmax(-1)
    return gt.reshape((h,w))

class TfObjectIdentifier:
  def __init__(self,existingModelFile,categoryFile,nBox):
    self._nBox = nBox
    self._modFile = existingModelFile
    self._catFile = categoryFile
    # this graph
    self._detection_graph = tf.Graph()
    with self._detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v2.io.gfile.GFile(self._modFile, 'rb') as fid:
        serialized_graph = fid.read()
        print(self._modFile)
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    f = open(self._catFile)
    catText = f.read()
    f.close()
    self._category_index = {}
    for entry in catText.split('item {')[1:]:
      idNum = int(entry.split('id:')[1].split('\n')[0].strip())
      idName = entry.split('name:')[1].split('\n')[0].strip()[1:-1]
      self._category_index[idNum] = {'id':idNum, 'name':idName}
    self._sess = tf.compat.v1.Session(graph=self._detection_graph)
    # for my own convenience
    self._numToName = {}
    for d in self._category_index.values():
      self._numToName[d['id']] = d['name']
  def getClassIds(self):
    outD = {}
    for d in self._category_index.values():
      outD[d['name']] = d['id']
    return outD
  def nBoxes(self): return self._nBox
  def getBoxes(self,image):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
    classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = self._sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
    h,w,ch = image.shape
    bL,scL,numB = boxes[0],scores[0],num_detections[0]
    classL = classes[0]
    boxL = []
    for n in range(int(numB)):
       yA,yB = int(bL[n][0]*h),int(bL[n][2]*h)
       xA,xB = int(bL[n][1]*w),int(bL[n][3]*w)
       clName = self._numToName[classL[n]]
       boxL.append( Box(xA,yA,xB,yB,scL[n],clName) )
    return boxL

class Box:
  def __init__(self,x0,y0,x1,y1,score,clName):
    self._x0, self._y0 = x0, y0
    self._x1, self._y1 = x1, y1
    self._score = score
    self._clName = clName
  # recover coords with min/max values
  def xMin(self): return min([self._x0,self._x1])
  def yMin(self): return min([self._y0,self._y1])
  def xMax(self): return max([self._x0,self._x1])
  def yMax(self): return max([self._y0,self._y1])
  def score(self): return self._score
  def name(self): return self._clName
  def exists(self):
    return self._x0 != self._x1 and self._y0 != self._y1
  # to allow for modifications
  def copy(self):
    return Box(self._x0,self._y0,self._x1,self._y1,
               self._score,self._clName)
  def translate(self,xTrans,yTrans):
    self._x0,self._x1 = self._x0 + xTrans, self._x1 + xTrans
    self._y0,self._y1 = self._y0 + yTrans, self._y1 + yTrans
  def constrain(self,imgW,imgH):
    if self.xMin() < 0:
      if self.xMax() < 0: self._x0,self._x1 = 0,0
      else: self._x0,self._x1 = 0,self.xMax()
    if self.yMin() < 0:
      if self.yMax() < 0: self._y0,self._y1 = 0,0
      else: self._y0,self._y1 = 0,self.yMax()
    if self.xMax() > imgW:
      if self.xMin() > imgW: self._x0,self._x1 = imgW,imgW
      else: self._x0,self._x1 = self.xMin(),imgW
    if self.yMax() > imgH:
      if self.yMin() > imgH: self._y0,self._y1 = imgH,imgH
      else: self._y0,self._y1 = self.yMin(),imgH



class TfClassifier:
  def __init__(self,existingModelFile,categoryFile):
    self._modFile = existingModelFile
    self._catFile = categoryFile
    proto_as_ascii_lines = tf.compat.v1.io.gfile.GFile(categoryFile).readlines()
    self._labels = list(map(lambda i: i.rstrip(), proto_as_ascii_lines))
    # ## Load a (frozen) Tensorflow model into memory.
    self._detection_graph = tf.Graph()
    with self._detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v1.io.gfile.GFile(self._modFile, 'rb') as fid:
        serialized_graph = fid.read()
        print(self._modFile)
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    self._sess = tf.compat.v1.Session(graph=self._detection_graph)
  def getClasses(self,image,spCl=None):
    # get the image tensor so I can re-size the image appropriately
    image_tensor = self._detection_graph.get_tensor_by_name('Placeholder:0')
    h,w = image.shape[:2]
    if h*w == 0:
      image = np.zeros(image_tensor.shape[1:])
    image_resized = cv2.resize(image,dsize=tuple(image_tensor.shape[1:3]))
    image_np_expanded = np.expand_dims(image_resized, axis=0)
    image_np_expanded = image_np_expanded.astype(np.float32)
    image_np_expanded /= 255
    answer_tensor = self._detection_graph.get_tensor_by_name('final_result:0')
    # Actual detection.
    (answer_tensor) = self._sess.run([answer_tensor],
                                     feed_dict={image_tensor: image_np_expanded})
    results = np.squeeze(answer_tensor)
    results = [(results[n],self._labels[n]) for n in range(len(self._labels))]
    return TfClassResult(results)
  def labels(self): return self._labels

class TfClassResult:
  # takes a list of score,label tuples
  def __init__(self,results):
    self._rD = {}
    for s,lb in results: self._rD[lb] = s
    self._lbmx = max(results)[1]
  def best(self): return self._lbmx
  def score(self,lb): return self._rD[lb]
  def labels(self): return self._rD.keys()

    

if __name__ == "__main__": main()
