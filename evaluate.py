import re


file_name='log_tmp/test_save/base_two_noarchi_save.log'

with open(file_name,'r') as fp:
    content=fp.read()

res_li=re.findall('train_acc:.*?([\d\.]+).*?eval_acc.*?([\d\.]+).*?test dataset acc.*?([\d\.]+).*?loss.*?([\d\.]+).*?auc_roc_score.*?([\d\.]+).*?',content)

print(len(res_li))

res=max(res_li,key=lambda x:float(x[1]))

print(f'train_acc:{res[0]}|eval_acc:{res[1]}|test_acc:{res[2]}|loss:{res[3]}|auc_roc:{res[4]}')