
import torch,os,cv2
import numpy as np
from models.net import net


os.environ['CUDA_VISIBLE_DEVICES']='0'
cuda = torch.cuda.is_available()

test_path = 'test_dir'
path = 'model.pth'

if __name__ == '__main__':

    refer_img = cv2.imread(os.path.join(test_path,'1.jpg'),0)
    refer_img = cv2.resize(refer_img,(220,155),cv2.INTER_LINEAR)
    refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])

    test_img = cv2.imread(os.path.join(test_path,'2.jpg'),0)
    test_img = cv2.resize(test_img,(220,155),cv2.INTER_LINEAR)
    test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])
    
    refer_test = np.concatenate((refer_img, test_img), axis=0)
    refer_test = torch.FloatTensor(refer_test)
    model = net()
    model.load_state_dict(torch.load(path))
    refer_test = refer_test.reshape(1,refer_test.shape[0],refer_test.shape[1],refer_test.shape[2])
    predict = model(refer_test)
    score = (predict[0]+predict[1]+predict[2])/3
    print(score)


