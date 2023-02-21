import torch

class Utilities:
    
    def __init__(self, pad_token = 38):
        self.pad_token = pad_token

    def ignore_pad_accuracy(self, preds, targets):
        
        preds_class = preds
        targets_class = targets
        
        ignore_mask = (~torch.eq(preds_class, self.pad_token)).type(torch.IntTensor)
        #print(f"The ignore mask is:{ignore_mask}")
        #print(torch.sum(ignore_mask))
        matches = (torch.eq(targets_class, preds_class)).type(torch.IntTensor) * ignore_mask
        #print(f"The correct matches are:{matches}")
        #print(torch.sum(matches))
        if  torch.sum(matches) == 0:
            ignore_accuracy = torch.tensor(0)
        else:
            ignore_accuracy = torch.sum(matches) / torch.sum(ignore_mask)
        #print(ignore_accuracy)
        return ignore_accuracy
    
    def remove_unneces(self, preds, length):
        length = length.item()
        #numb_list = [0] * length
        #k = 0
        # for i in range(len(preds)):
        #     if not(i > 0 and preds[i-1] == preds[i]):
        #         numb_list.insert(k, preds[i].item())
        #     k += 1
        #     if k == length:
        #         break
        numb_list = []
        for i in range(len(preds)):
            if not(i > 0 and preds[i-1] == preds[i]):
                numb_list.append(preds[i])
        numb_list = torch.tensor(numb_list)
        return numb_list
        
    # def reverce_tol(self, preds, tensor_length):
    #     """This function takes preds indices and  targets (padded)
    #     and returns the targets as non-padded and the appropriate
    #     preds tensor by dropping duplicates caused by the CTCLoss 
    #     as well as the blank character
    #     Args: 
    #         preds: The prediction indices
    #         targets: The targets (padded)
    #     Returns:
    #         preds_list: preds list with dropped blank and duplicates
    #         target_list: targets without padding"""
    #     length = tensor_length.item()
    #     preds_list = []
    #     k = 0
    #     while True:
    #         if k = (length - 1):
    #             break
    #         for i in range(len(preds)):
    #             if preds[i] != 0 and (not(i > 0 and preds[i-1] == preds[i])):
    #                 preds_list.insert(k, preds[i])
    #         return preds_list, target_list
        
        #remove a specific value from a tensor
        # targets_non_padded = targets[targets != 38]
        
        # preds_list = []
        # for i in range(len(preds)):