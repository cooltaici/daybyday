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

#暑期课题:在原图上裁剪一幅，对比
def img_feature_single_augment(path,                        #二级文件夹，包括良性和恶性子文件夹，存储的是裁剪过的图像，只是提供文件名
                               pathRow,                     #原始图像
                               pathMask,                    #一级文件夹，二值分割图像路径，良恶性放在一起，便于检索
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
            arrayi = np.zeros((perI*len(filename),target_size[0],target_size[1]))

        for k in range(len(filename)):
            cent = dictcent.get(filename[k])       #获取图像对应的裁剪信息
            ImgPath = pathRow + '\\' + filename[k]
            im = Image.open(ImgPath)
            if grayscale is True:
                im = im.convert("L")
            else:
                im = im.convert("RGB")        #可能是hsv模式，尽量改成RGB模式
            im = np.array(im)
            arrayi[(k*perI):((k+1)*perI)] = ImgAug(im,cent,target_size)   #把得到的扩增样本存储起来
        if i==0:
            sampleX = arrayi
            sampleY = labeli
        else:
            sampleX = np.vstack((sampleX,arrayi))
            sampleY = np.vstack((sampleY,labeli))
        del arrayi
    return sampleX/255.0,sampleY,main_filename

#暑期课题：在原图上裁剪一幅，对比
def img_feature_one(path,                        #二级文件夹，包括良性和恶性子文件夹，存储的是裁剪图像
                    pathRow,                     #原始图像
                    dictcent,                    #剪切范围，字典数据，包括文件名和剪切的内矩阵和外矩阵
                    grayscale=True,              #确定是灰度格式，还是RGB格式
                    marjin = 5,
                   target_size=(128,128)):       #图像规范化大小
    classpathlist = os.listdir(path)
    print(classpathlist)
    sampleX = np.array([])
    sampleY = np.array([])
    main_filename = list()
    for i in range(len(classpathlist)):
        classpath = path+'\\'+classpathlist[i]
        filename = os.listdir(classpath)
        main_filename = main_filename + filename
        print(classpath + ': ' + str(len(filename)) + ' samples')
        labeli = np.ones((len(filename),1))*i
        arrayi = np.zeros((len(filename),target_size[0],target_size[1]))
        if grayscale == False:
            arrayi = np.zeros((len(filename),target_size[0],target_size[1],3))

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
            ImgPath = pathRow + '\\' + filename[k]
            im = Image.open(ImgPath)

            if grayscale is True:
                im = im.convert("L")
            else:
                im = im.convert("RGB")                  #可能是hsv模式，尽量改成RGB模式
            im = np.array(im)

            Img = Image.fromarray(im[top:bottom,left:right])
            Img.save('E:\\数字图像处理\\tctest\\扩增测试文件夹\\'+filename[k])
            Img=Img.resize(target_size)
            im = np.array(Img)
            arrayi[k] = im
        if i==0:
            sampleX = arrayi
            sampleY = labeli
        else:
            sampleX = np.vstack((sampleX,arrayi))
            sampleY = np.vstack((sampleY,labeli))
        del arrayi
    return sampleX/255.0,sampleY,main_filename


#暑期课题：非ROI区域进行填充，填充一幅
def img_feature_one_fillb(  path,                        #二级文件夹，包括良性和恶性子文件夹，存储的是裁剪图像
                            grayscale=True,              #确定是灰度格式，还是RGB格式
                            target_size=(128,128)):       #图像规范化大小
    classpathlist = os.listdir(path)
    print(classpathlist)
    sampleX = np.array([])
    sampleY = np.array([])
    ratio = np.array([])
    main_filename = list()
    for i in range(len(classpathlist)):
        classpath = path+'\\'+classpathlist[i]
        filename = os.listdir(classpath)
        main_filename = main_filename + filename
        print(classpath + ': ' + str(len(filename)) + ' samples')
        ratioi = np.zeros((len(filename),1))
        labeli = np.ones((len(filename),1))*i
        arrayi = np.zeros((len(filename),target_size[0],target_size[1]))
        if grayscale == False:
            arrayi = np.zeros((len(filename),target_size[0],target_size[1],3))

        for k in range(len(filename)):
            ImgPath = classpath + '\\' + filename[k]
            im = Image.open(ImgPath)

            if grayscale is True:
                im = im.convert("L")
            else:
                im = im.convert("RGB")                  #可能是hsv模式，尽量改成RGB模式

            im = np.array(im)
            tmpImg = im
            if im.shape[0]>im.shape[1]:
                left = int(im.shape[0]/2.0 - im.shape[1]/2.0)
                ratioi[k] = target_size[0]/im.shape[0]
                if grayscale is False:
                    tmpImg = np.zeros((im.shape[0],im.shape[0],3))
                    tmpImg[:,left:(left+im.shape[1])] = im
                else:
                    tmpImg = np.zeros((im.shape[0],im.shape[0]))
                    tmpImg[:,left:(left+im.shape[1])] = im
            else:
                top = int(im.shape[1]/2.0 - im.shape[0]/2.0)
                ratioi[k] = target_size[0]/im.shape[1]
                if grayscale is False:
                    tmpImg = np.zeros((im.shape[1],im.shape[1],3))
                    tmpImg[top:(top+im.shape[0])] = im
                else:
                    tmpImg = np.zeros((im.shape[1],im.shape[1]))
                    tmpImg[top:(top+im.shape[0])] = im
            tmpImg = np.uint8(tmpImg)
            Img = Image.fromarray(tmpImg)
            #Img.save('E:\\数字图像处理\\tctest\\扩增测试文件夹\\'+filename[k])
            Img=Img.resize(target_size)
            im = np.array(Img)
            arrayi[k] = im
        if i==0:
            sampleX = arrayi
            sampleY = labeli
            ratio = ratioi
        else:
            sampleX = np.vstack((sampleX,arrayi))
            sampleY = np.vstack((sampleY,labeli))
            ratio = np.vstack((ratio,ratioi))
        del arrayi
    return sampleX/255.0,sampleY,ratio,main_filename

#暑期课题：非ROI区域进行填充，填充多幅，进行数据扩充
def img_feature_muL_fillb(  path,                        #二级文件夹，包括良性和恶性子文件夹，存储的是裁剪图像
                            times = 5,
                            grayscale = False,              #确定是灰度格式，还是RGB格式
                            target_size = (128,128)):       #图像规范化大小
    classpathlist = os.listdir(path)
    print(classpathlist)
    sampleX = np.array([])
    sampleY = np.array([])
    ratio = np.array([])
    main_filename = list()

    for i in range(len(classpathlist)):
        classpath = path+'\\'+classpathlist[i]
        filename = os.listdir(classpath)
        main_filename = main_filename + filename
        print(classpath + ': ' + str(len(filename)) + ' samples')
        labeli = np.ones((len(filename)*times,1))*i
        arrayi = np.zeros((len(filename)*times,target_size[0],target_size[1]))
        ratioi = np.zeros((len(filename)*times,1))
        if grayscale == False:
            arrayi = np.zeros((len(filename)*times,target_size[0],target_size[1],3))

        for k in range(len(filename)):
            ImgPath = classpath + '\\' + filename[k]
            im = Image.open(ImgPath)

            if grayscale is True:
                im = im.convert("L")
            else:
                im = im.convert("RGB")                  #可能是hsv模式，尽量改成RGB模式

            im = np.array(im)
            tmpImg = im
            tmpratio = 1
            #如果纵横比大于1
            if im.shape[0]>im.shape[1]:
                kstep = (im.shape[0]-im.shape[1])/(times-1)
                tmpratio = target_size[0]/im.shape[0]
                if grayscale is False:
                    tmpImg = np.zeros((times,im.shape[0],im.shape[0],3))
                    for ki in range(times):
                        left = int(kstep*ki)
                        tmpImg[ki,:,left:(left+im.shape[1])] = im
                else:
                    tmpImg = np.zeros((times,im.shape[0],im.shape[0]))
                    for ki in range(times):
                        left = int(kstep*ki)
                        tmpImg[ki,:,left:(left+im.shape[1])] = im
            #如果纵横比小于1
            else:
                kstep = (im.shape[1]-im.shape[0])/(times-1)
                tmpratio = target_size[0]/im.shape[1]
                if grayscale is False:
                    tmpImg = np.zeros((times,im.shape[1],im.shape[1],3))
                    for ki in range(times):
                        top = int(kstep*ki)
                        tmpImg[ki,top:(top+im.shape[0])] = im
                else:
                    tmpImg = np.zeros((times,im.shape[1],im.shape[1]))
                    for ki in range(times):
                        top = int(kstep*ki)
                        tmpImg[ki,top:(top+im.shape[0])] = im

            tmpImg = np.uint8(tmpImg)
            for ki in range(times):
                Img = Image.fromarray(tmpImg[ki])
                #Img.save('E:\\数字图像处理\\tctest\\扩增测试文件夹\\'+filename[k])
                Img=Img.resize(target_size)
                im = np.array(Img)
                arrayi[k*times+ki] = im
            ratioi[(k*times):((k+1)*times)] = tmpratio
        if i==0:
            sampleX = arrayi
            sampleY = labeli
            ratio = ratioi
        else:
            sampleX = np.vstack((sampleX,arrayi))
            sampleY = np.vstack((sampleY,labeli))
            ratio = np.vstack((ratio,ratioi))
        del arrayi,ratioi
    return sampleX/255.0,sampleY,ratio,main_filename

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

#将最小裁剪 变成正方形裁剪
def img_Center_Crop(pathImg,
                    pathCenter,
                    pathCroped,
                    pathSave,
                    margin = 0,
                    grayscale = False):
    Imgpathlist=os.listdir(pathImg)
    Croplist = os.listdir(pathCroped)
    str = open(pathCenter,'r').read()
    ncount = len(Imgpathlist)
    #有些图像里可能有多个肿瘤，这个时候以裁剪过的图像数目为准
    for i in range(len(Croplist)):
        # if i==75:
        #     print(Imgpathlist[i])
        index = str.find('\n')
        if index == -1:
            continue
        temp = str[0:index+1]   #这里改了一下会不会有影响
        str = str[index+1:]
        index = temp.find('.')
        #picture ID
        IDstr = temp[0:index]
        for j in range(len(Croplist)):
            index3 = Croplist[j].find('.')
            strlistt = Croplist[j][0:index3]
            if strlistt== IDstr:
                #打开原始图像，这里因为存在副本所以要处理一下
                indexE = Croplist[j].find(" - 副本")
                tempOrig = Croplist[j]
                if indexE != -1:
                    tempOrig = Croplist[j][0:indexE]
                    indexE = Croplist[j].find(".")
                    tempOrig = tempOrig + Croplist[j][indexE:len(Croplist[j])]
                if os.path.exists(pathImg+'\\'+tempOrig) == False:
                    break
                im = Image.open(pathImg+'\\'+tempOrig)
                if grayscale == True:
                   im = im.convert("L")
                #image to numpy
                im = np.array(im)
                #position；找出裁剪图像的中心点（x0，x1）,有的图像命名含有括号，所以如下所示
                index0 = temp.find(':(')
                index1 = temp.find(',')
                x0 = int(temp[index0+2:index1])
                index0 = temp.find(')\n')
                x1 = int(temp[index1+1:index0])
                #裁剪图像
                cropstr = pathCroped + '\\'+ Croplist[j]
                if os.path.exists(cropstr):
                   imcrop = Image.open(cropstr)
                maxpix = max(imcrop.size[0]+margin*2,imcrop.size[1]+margin*2)

                left = int(x0 - maxpix/2)
                right = int(x0 + maxpix/2 + 1)
                top = int(x1 - maxpix/2)
                bottom = int(x1 + maxpix/2 + 1)

                if x0-maxpix/2.0 < 0:
                    left = 0
                    right = maxpix
                if x0+maxpix/2.0 > im.shape[1]:
                    right = im.shape[1]
                    left = im.shape[1] - maxpix
                if x1-maxpix/2.0 < 0:
                    top = 0
                    bottom = maxpix
                if x1+maxpix/2.0 > im.shape[0]:
                    bottom = im.shape[0]
                    top = im.shape[0] - maxpix
                    if top<0:
                        top = 0
                        left = x0 - int(im.shape[0]/2)
                        right =left + im.shape[0]
                im = im[top:bottom,left:right,:]

                #将numpy数组转化为Image图像并存储
                imgsave = Image.fromarray(im)
                if os.path.exists(pathSave) == False:
                    os.makedirs(pathSave)
                #print(Imgpathlist[j])
                imgsave.save(pathSave+'\\'+Croplist[j])
                break
        if j == len(Croplist)-1:
            print(Croplist[j])