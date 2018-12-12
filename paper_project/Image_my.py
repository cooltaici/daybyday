# -*- coding: utf-8 -*-
#image processing
#from keras.preprocessing.image import array_to_img,img_to_array,load_img
import numpy as np
import os,shutil
from PIL import Image
import h5py,pickle
import pickle
import matplotlib.pyplot as plt
#%matplotlib inline
#根据文件夹，生成标签数据
def generate_label(path, isave=False):
    pathlist = os.listdir(path)
    featureDict = dict()
    for k in range(len(pathlist)):
        tmpath = path+'\\'+pathlist[k]
        Imglist = os.listdir(tmpath)
        for m in range(len(Imglist)):
            featureDict[Imglist[m]] = list([k])
    if isave==True:
        savepath = path+'label.pkl'
        f = open(savepath,'wb')
        pickle.dump(featureDict,f)
        f.close()
    return featureDict

#y移动错分的样本
def remove_wrong(pathtxt,pathsource,pathdest,pathCorrect,listt):
    pathlist1 = os.listdir(pathsource)
    pathlist2 = os.listdir(pathdest)
    pathlist3 = os.listdir(pathCorrect)
    wsample = open(pathtxt,'r').read()
    dorun = True
    while dorun:
        index = wsample.find('\n')
        if index==-1:
            break
        tmpstr = wsample[0:index]
        wsample = wsample[index+1:]
        index = tmpstr.find(' :')
        ImgID = tmpstr[0:index]
        targetIndex = int(tmpstr[index+3])
        #label = int(tmpstr[index+1])
        for i in range(len(pathlist1)):
            paths = pathsource + '\\' + pathlist1[i]+ '\\'+ImgID
            pathd = pathdest + '\\' + pathlist2[i]+ '\\'+ImgID
            pathc = pathCorrect + '\\' + pathlist3[listt[targetIndex]]+ '\\'+ImgID
            if os.path.exists(paths):
                shutil.move(paths,pathd)
                shutil.copy(pathd,pathc)

#手动给图像赋予标签，生成特征数据
def feature_from_files(pathmain,pathfeature,pathtxt = None):
    pathlist0 = os.listdir(pathmain)
    pathlist1 = os.listdir(pathfeature)
    filename = list()
    featurelist = list()
    #labelist = list()
    for k in range(len(pathlist0)):  #
        tmpath = pathmain + '\\' + pathlist0[k]
        mainlist = os.listdir(tmpath)
        for m in range(len(mainlist)):         #main库里的每一幅图像
            tmplist = list()
            tmplist.append(k)                   #添加标签
            filename.append(mainlist[m])        #添加图像ID
            isremove = False
            for k1 in range(len(pathlist1)):   #各个特征文件夹
                tmpath1 = pathfeature  + '\\' +pathlist1[k1]
                felist = os.listdir(tmpath1)
                for m1 in range(len(felist)):   #特征文件夹子文件夹
                    tmpath2 = pathfeature  + '\\' +pathlist1[k1]+'\\'+felist[m1]+'\\'+mainlist[m]
                    if os.path.exists(tmpath2):
                        tmplist.append(m1)
                        break
                    if m1 == len(felist)-1:
                        isremove = True
                if isremove:
                    filename.pop()
                    break
            if isremove == False:
               featurelist.append(tmplist)
    nfe = len(featurelist[0])
    fenp = np.zeros((len(featurelist),nfe))
    for k in range(fenp.shape[0]):
        fenp[k,:] = np.array(featurelist[k])
    if pathtxt != None:
        pathfeature = pathtxt + '\\' + 'feature.txt'
        pathfile = pathtxt + '\\' + 'filename.txt'
        f = open(pathfile,'w')
        for k in range(len(filename)):
            f.write(filename[k]+'\n')
        f.close()
        f = open(pathfeature,'w')
        np.savetxt(pathfeature,fenp)
        f.close()
    return fenp,filename

#从文本文件中获取特征数据，存储到一个字典结构中
def Txt2data_feture(pathtxt):
    fullstr = open(pathtxt,'r').read()
    dict_fe=dict()
    dorun = True
    while dorun:
        index = fullstr.find('\n\n')
        if index == -1:
            break
        tmpstr = fullstr[0:index]
        fullstr = fullstr[index+2:]
        tmpfe = np.zeros((1,5))
        index0 = tmpstr.find(':')
        filename = tmpstr[0:index0]
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpfe[0,0] = float(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpfe[0,1] = float(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpfe[0,2] = float(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpfe[0,3] = float(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        tmpfe[0,4] = float(tmpstr)
        dict_fe[filename] = tmpfe
    return dict_fe



#根据剪切信息，自己编的图像扩增算法
def img_single_augment(path,                        #一级文件夹，包括良性和恶性子文件夹，存储的是原始未裁剪图像
                       dictcent,                    #剪切范围，字典数据，包括文件名和剪切的内矩阵和外矩阵
                       grayscale=True,              #确定是灰度格式，还是RGB格式
                       times = 4,
                       target_size=(128,128)):       #图像规范化大小
    classpathlist = os.listdir(path)
    print(classpathlist)
    sampleX = np.array([])
    sampleY = np.array([])
    perI = times*times*times
    main_filename = list()
    for i in range(len(classpathlist)):
        classpath = path+'\\'+classpathlist[i]
        filename = os.listdir(classpath)
        main_filename = main_filename + filename
        print(classpath + ':' + str(len(filename)) + 'samples')
        labeli = np.ones((perI*len(filename),1))*i
        arrayi = np.zeros((perI*len(filename),target_size[0],target_size[1]))
        if grayscale==False:
            arrayi = np.zeros((perI*len(filename),target_size[0],target_size[1],3))

        for k in range(len(filename)):
            cent = dictcent.get(filename[k])       #获取图像对应的裁剪信息
            ImgPath=classpath + '\\' + filename[k]
            im = Image.open(ImgPath)
            if grayscale is True:
                im = im.convert("L")
            else:
                im = im.convert("RGB")        #可能是hsv模式，尽量改成RGB模式
            im = np.array(im)
            arrayi[(k*perI):((k+1)*perI)] = ImgAug(im,cent,target_size)
            #arrayi[(k*perI):((k+1)*perI)] = ImgAug(im,cent,target_size,isave=True,saveImgpath='D:\\tctest\\扩增测试',filename=filename[k])   #把得到的扩增样本存储起来
        if i==0:
            sampleX = arrayi
            sampleY = labeli
        else:
            sampleX = np.vstack((sampleX,arrayi))
            sampleY = np.vstack((sampleY,labeli))
        del arrayi
    return sampleX/255.0,sampleY,main_filename

#根据剪切信息，自己编的图像（以及对应的二值化分割图像）扩增算法
def img_mask_single_augment(path,                        #二级文件夹，包括良性和恶性子文件夹，存储的是原始未裁剪图像
                            pathMask,                    #一级文件夹，二值分割图像路径，良恶性放在一起，便于检索
                            dictcent,                    #剪切范围，字典数据，包括文件名和剪切的内矩阵和外矩阵
                            grayscale=True,              #确定是灰度格式，还是RGB格式
                            times = 4,
                            target_size=(128,128)):       #图像规范化大小
    classpathlist = os.listdir(path)
    print(classpathlist)
    sampleX = np.array([])
    sampleM = np.array([])
    sampleY = np.array([])
    perI = times*times*times
    main_filename = list()
    for i in range(len(classpathlist)):
        classpath = path+'\\'+classpathlist[i]
        filename = os.listdir(classpath)
        main_filename = main_filename + filename
        print(classpath + ':' + str(len(filename)) + 'samples')
        labeli = np.ones((perI*len(filename),1))*i
        arrayi = np.zeros((perI*len(filename),target_size[0],target_size[1]))
        maski = np.zeros((perI*len(filename),target_size[0],target_size[1]))
        if grayscale==False:
            arrayi = np.zeros((perI*len(filename),target_size[0],target_size[1]))

        for k in range(len(filename)):
            cent = dictcent.get(filename[k])       #获取图像对应的裁剪信息
            ImgPath = classpath + '\\' + filename[k]
            maskPath = pathMask + '\\' +filename[k]
            im = Image.open(ImgPath)
            mask = Image.open(maskPath)
            if grayscale is True:
                im = im.convert("L")
            else:
                im = im.convert("RGB")        #可能是hsv模式，尽量改成RGB模式
            im = np.array(im)
            mask = np.array(mask)
            arrayi[(k*perI):((k+1)*perI)] = ImgAug(im,cent,target_size)   #把得到的扩增样本存储起来
            maski[(k*perI):((k+1)*perI)] = MaskAug(mask,im,cent,target_size)
        if i==0:
            sampleX = arrayi
            sampleY = labeli
            sampleM  = maski
        else:
            sampleX = np.vstack((sampleX,arrayi))
            sampleY = np.vstack((sampleY,labeli))
            sampleM = np.vstack((sampleM,maski))
        del arrayi,maski
    return sampleX/255.0,sampleM/255.0,sampleY,main_filename

#根据剪切信息，自己编的图像（以及对应的二值化分割图像）扩增算法
def img_mask_one_augment(path,                        #二级文件夹，包括良性和恶性子文件夹，存储的是原始未裁剪图像
                         pathMask,                    #一级文件夹，二值分割图像路径，良恶性放在一起，便于检索
                         dictcent,                    #剪切范围，字典数据，包括文件名和剪切的内矩阵和外矩阵
                         grayscale=True,              #确定是灰度格式，还是RGB格式
                         marjin = 10,
                         target_size=(128,128)):       #图像规范化大小
    classpathlist = os.listdir(path)
    print(classpathlist)
    sampleX = np.array([])
    sampleM = np.array([])
    sampleY = np.array([])
    main_filename = list()
    for i in range(len(classpathlist)):
        classpath = path+'\\'+classpathlist[i]
        filename = os.listdir(classpath)
        main_filename = main_filename + filename
        print(classpath + ':' + str(len(filename)) + 'samples')
        labeli = np.ones((len(filename),1))*i
        arrayi = np.zeros((len(filename),target_size[0],target_size[1]))
        if grayscale == False:
            arrayi = np.zeros((len(filename),target_size[0],target_size[1],3))
        maski = np.zeros((len(filename),target_size[0],target_size[1]))

        for k in range(len(filename)):
            cent = dictcent.get(filename[k])             #获取图像对应的裁剪信息
            #确定裁剪区域
            left=right=top=bottom=0
            if cent[1]-cent[0]<cent[3]-cent[2]:               #如果宽大于高
                left = int(cent[2]-marjin)
                right = int(cent[3]+marjin)
                top = int((cent[0]+cent[1])/2 - (right-left)/2+0.5)
                bottom = int((cent[0]+cent[1])/2 + (right-left)/2+0.5)
                if bottom>cent[5]:
                    bottom = int(cent[5])
                    top = int(bottom - (right-left))
                else:
                    top = int(bottom - (right-left))
                if top<cent[4]:
                    top = int(cent[4])
                    bottom = int(top + (right-left))
                else:
                    bottom = int(top + (right-left))
            else:                                           #如果高大于宽
                top = int(cent[0]-marjin)
                bottom = int(cent[1]+marjin)
                left = int((cent[2]+cent[3])/2 - (bottom-top)/2+0.5)
                right = int((cent[2]+cent[3])/2 + (bottom-top)/2+0.5)
                if left<cent[6]:
                    left = int(cent[6])
                    right = int(left + (bottom-top))
                else:
                    right = int(left + (bottom-top))
                if right>cent[7]:
                    right = int(cent[7])
                    left = int(right - (bottom - top))
                else:
                    left = int(right - (bottom - top))
            #
            ImgPath = classpath + '\\' + filename[k]
            maskPath = pathMask + '\\' +filename[k]
            im = Image.open(ImgPath)
            mask = Image.open(maskPath)

            if grayscale is True:
                im = im.convert("L")
            else:
                im = im.convert("RGB")                  #可能是hsv模式，尽量改成RGB模式
            im = np.array(im)

            mask = np.array(mask)
            tmpMask = np.zeros((im.shape[0],im.shape[1]))
            tmpMask[int(cent[0]):int(cent[0]+mask.shape[0]),int(cent[2]):int(cent[2]+mask.shape[1])] = mask

            Img = Image.fromarray(im[top:bottom,left:right])
            #Img.save('E:\\图像数据库\\乳腺肿瘤数据集-tc版\\benign_63正方形裁剪\\'+filename[k])
            Img=Img.resize(target_size)

            ImMask = Image.fromarray(tmpMask[top:bottom,left:right])
            ImMask = ImMask.resize(target_size)
            ImMask = ImMask.convert('L')

            #ImMask.save('E:\\图像数据库\\乳腺肿瘤数据集-tc版\\malignan_正方形裁剪\\'+filename[k])
            im = np.array(Img)
            tmpMask = np.array(ImMask)
            arrayi[k] = im
            maski[k] = tmpMask
        if i==0:
            sampleX = arrayi
            sampleY = labeli
            sampleM  = maski
        else:
            sampleX = np.vstack((sampleX,arrayi))
            sampleY = np.vstack((sampleY,labeli))
            sampleM = np.vstack((sampleM,maski))
        del arrayi,maski
    return sampleX/255.0,sampleM/255.0,sampleY,main_filename

#数据扩增
def Text2dick(pathtxt):
    fullstr = open(pathtxt,'r').read()
    dict_center=dict()
    dorun = True

    #根据文本文件，生成裁剪的字典数据
    while dorun:
        index = fullstr.find('\n\n')
        if index == -1:
            break
        tmpcen = np.zeros(8)
        tmpstr = fullstr[0:index]
        fullstr = fullstr[index+2:]
        index0 = tmpstr.find(':')
        filename = tmpstr[0:index0]
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpcen[0] = int(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpcen[1] = int(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpcen[2] = int(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpcen[3] = int(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]

        index0 = tmpstr.find(' ')
        tmpcen[4] = int(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpcen[5] = int(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        index0 = tmpstr.find(' ')
        tmpcen[6] = int(tmpstr[0:index0])   #
        tmpstr = tmpstr[index0+1:]
        tmpcen[7] = int(tmpstr)

        dict_center[filename] = tmpcen
    #跟据字典文件进行数据扩增:
    return dict_center

#对数据进行扩增，Img为Numpy数组，te为裁剪矩形的坐标和外边界矩形的坐标
def ImgAug(Img,te,tsize,isave = False,saveImgpath = None,filename = None,times = 4):
    #确定偏移的外边界
    Imgau = []
    if len(Img.shape)==2:
        Imgau = np.zeros((times*times*times,tsize[0],tsize[1]))
    if len(Img.shape)==3:
        Imgau = np.zeros((times*times*times,tsize[0],tsize[1],3))
    outtop = outbot = outleft = outright = 0
    if te[0]-te[4]>=te[1]-te[0]:
        outtop = te[0]-(te[1]-te[0])
    else:
        outtop = te[4]
    if te[2]-te[6]>=te[3]-te[2]:
        outleft = te[2]-(te[3]-te[2])
    else:
        outleft = te[6]
    if te[5]-te[1]>=te[1]-te[0]:
        outbot = te[1]+(te[1]-te[0])
    else:
        outbot = te[5]
    if te[7]-te[3]>=te[3]-te[2]:
        outright = te[3]+(te[3]-te[2])
    else:
        outright = te[7]
    kt = (te[0]-outtop)/times
    kb = (outbot-te[1])/times
    kl = (te[2]-outleft)/times
    kr = (outright-te[3])/times
    Imgth = 0
    for i in range(times):
        intop = int(te[0] - i*kt+0.5)                     #固定左上角行
        for j in range(times):
            inleft = int(te[2] - j*kl+0.5)                #固定左上角列
            if te[3]-inleft>=te[1]-intop:
                for k in range(times):
                    inright = int(te[3]+kr*k+0.5)
                    inbot = intop+(inright-inleft)
                    if inright-inleft>te[5]-intop:
                        inbot = te[5]
                        inright = inleft + (te[5]-intop)
                    inright = int(inright)
                    inbot = int(inbot)
                    intop = int(intop)
                    inleft = int(inleft)
                    #裁剪图像
                    tmpcrop = Img[intop:inbot,inleft:inright]
                    newImg = Image.fromarray(tmpcrop)
                    newImg = newImg.resize((tsize[0],tsize[1]))
                    tmpnp = np.array(newImg)
                    Imgau[Imgth] = tmpnp

                    Imgth = Imgth+1
                    #存储扩增的图片
                    if isave==True:
                        index2 = filename.find('.')
                        filename1 = filename[0:index2]
                        filename1 = filename1 + '_' + str(i) + str(j) + str(k)
                        newImg.save(saveImgpath + '\\' + filename1 + '.jpg')
            else:
                for k in range(times):
                    inbot = int(te[1]+kb*k+0.5)
                    inright = inleft+(inbot-intop)
                    if inbot-intop>inright-te[2]:
                        inright = te[7]                    #这里有问题！
                        inbot = intop + (te[7]-inleft)
                    inright = int(inright)
                    inbot = int(inbot)
                    intop = int(intop)
                    inleft = int(inleft)
                    #裁剪图像
                    tmpcrop = Img[intop:inbot,inleft:inright]
                    newImg = Image.fromarray(tmpcrop)
                    newImg = newImg.resize((tsize[0],tsize[1]))
                    tmpnp = np.array(newImg)
                    Imgau[Imgth] = tmpnp
                    Imgth = Imgth+1
                    #存储扩增的图片
                    if isave==True:
                        index2 = filename.find('.')
                        filename1 = filename[0:index2]
                        filename1 = filename1 + '_' + str(i) + str(j) + str(k)
                        newImg.save(saveImgpath + '\\' + filename1 + '.jpg')
    return Imgau

#分割二值化图像的扩增。Mask:裁剪后的二值图像。Img：原始图像。te：裁剪信息。tsize：扩增目标大小。filename:存储的文件名。
def MaskAug(Mask,Img,te,tsize,isave = False,saveImgpath = None,filename = None,times = 4):
    #补充Mask,使得大小和Img一样大
    tmpMask = np.zeros((Img.shape[0],Img.shape[1]))
    tmpMask[int(te[0]):int(te[0]+Mask.shape[0]),int(te[2]):int(te[2]+Mask.shape[1])] = Mask
    Mask = tmpMask
    del tmpMask
    Imgau = []
    if len(Mask.shape)==2:
        Imgau = np.zeros((times*times*times,tsize[0],tsize[1]))
    if len(Mask.shape)==3:
        Imgau = np.zeros((times*times*times,tsize[0],tsize[1],3))
    #确定偏移的外边界
    outtop = outbot = outleft = outright = 0
    if te[0]-te[4]>=te[1]-te[0]:
        outtop = te[0]-(te[1]-te[0])
    else:
        outtop = te[4]
    if te[2]-te[6]>=te[3]-te[2]:
        outleft = te[2]-(te[3]-te[2])
    else:
        outleft = te[6]
    if te[5]-te[1]>=te[1]-te[0]:
        outbot = te[1]+(te[1]-te[0])
    else:
        outbot = te[5]
    if te[7]-te[3]>=te[3]-te[2]:
        outright = te[3]+(te[3]-te[2])
    else:
        outright = te[7]
    kt = (te[0]-outtop)/times
    kb = (outbot-te[1])/times
    kl = (te[2]-outleft)/times
    kr = (outright-te[3])/times
    Imgth = 0
    for i in range(times):
        intop = int(te[0] - i*kt+0.5)                     #固定左上角行
        for j in range(times):
            inleft = int(te[2] - j*kl+0.5)                #固定左上角列
            if te[3]-inleft>=te[1]-intop:
                for k in range(times):
                    inright = int(te[3]+kr*k+0.5)
                    inbot = intop+(inright-inleft)
                    if inright-inleft>te[5]-intop:
                        inbot = te[5]
                        inright = inleft + (te[5]-intop)
                    inright = int(inright)
                    inbot = int(inbot)
                    intop = int(intop)
                    inleft = int(inleft)
                    #裁剪图像
                    tmpcrop = Mask[intop:inbot,inleft:inright]
                    newImg = Image.fromarray(tmpcrop)
                    newImg = newImg.resize((tsize[0],tsize[1]))
                    newImg = newImg.convert('L')
                    tmpnp = np.array(newImg)
                    Imgau[Imgth] = tmpnp

                    Imgth = Imgth+1
                    #存储扩增的图片
                    if isave==True:
                        index2 = filename.find('.')
                        filename1 = filename[0:index2]
                        filename1 = filename1 + '_' + str(i) + str(j) + str(k)
                        newImg.save(saveImgpath + '\\' + filename1 + '.jpg')
            else:
                for k in range(times):
                    inbot = int(te[1]+kb*k+0.5)
                    inright = inleft+(inbot-intop)
                    if inbot-intop>inright-te[2]:
                        inright = te[7]
                        inbot = intop + (te[7]-inleft)
                    inright = int(inright)
                    inbot = int(inbot)
                    intop = int(intop)
                    inleft = int(inleft)
                    #裁剪图像
                    tmpcrop = Mask[intop:inbot,inleft:inright]
                    newImg = Image.fromarray(tmpcrop)
                    newImg = newImg.resize((tsize[0],tsize[1]))
                    newImg = newImg.convert('L')
                    tmpnp = np.array(newImg)
                    Imgau[Imgth] = tmpnp
                    Imgth = Imgth+1
                    #存储扩增的图片
                    if isave==True:
                        index2 = filename.find('.')
                        filename1 = filename[0:index2]
                        filename1 = filename1 + '_' + str(i) + str(j) + str(k)
                        newImg.save(saveImgpath + '\\' + filename1 + '.jpg')
    return Imgau


#图像数据集 交叉验证 数据处理
def Img_Split_Cross(pathOrig,
                    pathdest,
                    nCross = 5,
                    randSeed = None):
    '''generate cross_validate dataset with given directory
    # Arguments
        pathOrig:original image path, if there are 3 class, then there need to be 3 sub_file under the path
        pathdest:the output file path you want
        nCross: hao many cross you want. If you are doing 5-cross-validation, one fold is used to as test_data, and others are used as train_data
        randSeed:a ranmdom sequence for shuffle data in your order
    '''
    if os.path.exists(pathdest) == False:
        os.makedirs(pathdest)

    origClass = os.listdir(pathOrig)
    nclass = len(origClass)
    for nc in range(nclass):
        ncimglist = os.listdir(pathOrig+"\\"+origClass[nc])
        ncount = len(ncimglist)
        randIndex = np.arange(ncount)
        np.random.shuffle(randIndex)

        # if randSeed != None:
        #     randIndex = randSeed
        #     if randSeed.size != ncount:
        #         return -1

        avage = int(ncount/nCross)
        left = np.mod(ncount,nCross)
        #把图像分成nCross存进每个目录下面
        startindex = 0
        for i in range(nCross):
            #建立所有子目录
            for j in range(nCross):
                strfile = "\\Cross%d"%j
                if os.path.exists(pathdest+strfile) == False:
                    os.makedirs(pathdest+strfile)
                if os.path.exists(pathdest+strfile+"\\"+"train\\"+origClass[nc]) == False:
                    os.makedirs(pathdest+strfile+"\\"+"train\\"+origClass[nc])
                if os.path.exists(pathdest+strfile+"\\"+"test\\"+origClass[nc]) == False:
                    os.makedirs(pathdest+strfile+"\\"+"test\\"+origClass[nc])
            #每一幅都存几个目录
            sample0 = avage
            if i < left:
                sample0 = sample0+1
            for im in range(startindex,startindex+sample0):
                #读取原图像
                img = Image.open(pathOrig+"\\"+origClass[nc]+"\\"+ncimglist[randIndex[im]])
                #存储图像
                for k in range(nCross):
                    ttstr = "Cross%d"%k
                    if k==i:
                        img.save(pathdest+"\\"+ttstr+"\\"+"test"+"\\"+origClass[nc]+"\\"+ncimglist[randIndex[im]])
                    else:
                        img.save(pathdest+"\\"+ttstr+"\\"+"train"+"\\"+origClass[nc]+"\\"+ncimglist[randIndex[im]])
            startindex = sample0 + startindex

#单目录下的图像读程序,该目录下有几个子文件夹，那么就有多少类别，格式为n*x*x
def img_single_directory(path,
                         grayscale=True,
                         target_size=(28,28),
                         shuffle=True):
    '''generate dataset under the given directory
    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: normalize the dataset with the given size
        shuffle: if it is ture,then it would be shuffled
    '''
    classpathlist=os.listdir(path)
    print(classpathlist)
    sampX=np.array([])
    sampY=np.array([])
    filename_all = list()
    for i in np.arange(len(classpathlist)):
        print('directory:' +classpathlist[i])
        tempath=path+'\\'+classpathlist[i]
        templist=os.listdir(tempath)
        filename_all = filename_all + templist

        print(len(templist))
        arrayj=np.zeros((len(templist),target_size[0],target_size[1]))
        if grayscale==False:
            arrayj=np.zeros((len(templist),target_size[0],target_size[1],3))

        label=np.ones((len(templist),1))*i

        for j in np.arange(len(templist)):
            picturePath=tempath+'\\'+templist[j]
            #list.append(templist[j])
            im = Image.open(picturePath)
            if grayscale is True:
                im = im.convert("L")
            else:
                im = im.convert("RGB")        #可能是hsv模式，尽量改成RGB模式
            im = im.resize(target_size)
            arrayj[j] = np.array(im)

        if sampX.size is 0:
            sampX=arrayj.copy()
        else:
            sampX=np.vstack((sampX,arrayj))
        if sampY.size is 0:
            sampY=label.copy()
        else:
            sampY=np.vstack((sampY,label))
    #shuffle the dataset if necessory!we can shuffle twice here!
    if shuffle is True:
        #index=range(sampX.shape[0])
        index=np.arange(sampX.shape[0]).reshape(sampX.shape[0],-1)
        np.random.shuffle(index)
        index = index.reshape(-1)
        sampX=sampX[index]
        sampY=sampY[index]
    return sampX/255.0,sampY,filename_all

#单目录下的图像读程序,该目录下没有子文件夹，那么就有多少类别
def img_read_directory(path,
                       grayscale = False,
                       target_size=(224,224)):
    '''generate dataset under the given directory
    # Arguments
        path: path to image file
        target_size: normalize the dataset with the given size
    '''
    imagelist = os.listdir(path)

    sampleX = np.zeros((len(imagelist),target_size[0],target_size[1]))
    if grayscale==False:
        sampleX = np.zeros((len(imagelist),target_size[0],target_size[1],3))

    print('samples:',len(imagelist))
    for k in range(len(imagelist)):
        tmpth = path + '\\' + imagelist[k]
        im = Image.open(tmpth)
        if grayscale is True:
            im = im.convert("L")
        else:
            im = im.convert("RGB")        #可能是hsv模式，尽量改成RGB模式
        im = im.resize(target_size)
        sampleX[k] = np.array(im)
    return sampleX/255.0,imagelist

#根据文件夹中的图片，特征文本文件，生成特征矩阵和对应的标签
def feature_single_directory(path,                        #一级文件夹，包括良性和恶性子文件夹，存储的是原始未裁剪图像
                             pathdict,                    #特征文本路径
                             nfe):                        #特征数目
    fedict = Txt2data_feture(pathdict)
    classpathlist = os.listdir(path)
    print(classpathlist)
    sampleX = np.array([])
    sampleY = np.array([])
    main_filename = list()
    for i in range(len(classpathlist)):
        classpath = path+'\\'+classpathlist[i]
        filename = os.listdir(classpath)
        main_filename = main_filename + filename
        print(classpath + ':' + str(len(filename)) + 'samples')
        labeli = np.ones((len(filename),1))*i
        arrayi = np.zeros((len(filename),nfe))
        for k in range(len(filename)):
            fek = fedict.get(filename[k])       #获取图像对应的裁剪信息
            arrayi[k] = fek[0,0:nfe]
        if i==0:
            sampleX = arrayi
            sampleY = labeli
        else:
            sampleX = np.vstack((sampleX,arrayi))
            sampleY = np.vstack((sampleY,labeli))
    return sampleX,sampleY,main_filename

