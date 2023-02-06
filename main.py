import utility

if __name__ == '__main__':
    while True:
        datasetName = input()
        if datasetName == "-1":
            break
        scale = [2,3,4]
        original_path = "original/"+datasetName

        for s in scale:
            lr_path = "lr/"+datasetName+"/x"+str(s)
            sr_path = "sr/"+datasetName+"/x"+str(s)
            bicubic_path = "bicubic/"+datasetName+"/x"+str(s)
            sr, bi = utility.getPSNR(original_path,sr_path,bicubic_path,lr_path,s)

            print(f"{datasetName}-X{s}-- SR: {sr} , Bicubic: {bi}")

