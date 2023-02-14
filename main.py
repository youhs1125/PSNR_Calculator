import utility

if __name__ == '__main__':
    while True:
        print("Enter dataset Name (enter -1 to exit)")
        datasetName = input()
        if datasetName == "-1":
            break
        print("Does bicubic Files exist? (y/n)")
        isfileexist = input()

        scale = [2,3,4]
        original_path = "original/"+datasetName

        for s in scale:
            lr_path = "lr/"+datasetName+"/x"+str(s)
            sr_path = "sr/"+datasetName+"/x"+str(s)
            bicubic_path = "bicubic/"+datasetName+"/x"+str(s)
            sr, bi, sr_SSIM, bi_SSIM = utility.getPSNRnSSIM(original_path,sr_path,bicubic_path,lr_path,s,isfileexist)
            sr = round(sr,3)
            bi = round(bi,3)
            sr_SSIM = round(sr_SSIM,4)
            bi_SSIM = round(bi_SSIM,4)
            print(f"{datasetName}-X{s}-- (PSNR) SR: {sr} , Bicubic: {bi} \n (SSIM) SR: {sr_SSIM}, Bicubic: {bi_SSIM}")

