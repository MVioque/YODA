import numpy as np
import math
import csv
import time
import pandas as pd


start_time = time.time()



with open('Output_Input_sample_v3_good.csv', 'w') as out1, open('Output_Input_sample_v30.csv', 'r') as inp0, open('Output_Input_sample_v31.csv', 'r') as inp1, open('Output_Input_sample_v32.csv', 'r') as inp2, open('Output_Input_sample_v33.csv', 'r') as inp3, open('Output_Input_sample_v34.csv', 'r') as inp4, open('Output_Input_sample_v35.csv', 'r') as inp5, open('Output_Input_sample_v36.csv', 'r') as inp6, open('Output_Input_sample_v37.csv', 'r') as inp7, open('Output_Input_sample_v38.csv', 'r') as inp8, open('Output_Input_sample_v39.csv', 'r') as inp9:
    csvreader0 = csv.reader(inp0)
    csvreader1 = csv.reader(inp1)
    csvreader2 = csv.reader(inp2)
    csvreader3 = csv.reader(inp3)
    csvreader4 = csv.reader(inp4)
    csvreader5 = csv.reader(inp5)
    csvreader6 = csv.reader(inp6)
    csvreader7 = csv.reader(inp7)
    csvreader8 = csv.reader(inp8)
    csvreader9 = csv.reader(inp9) 
    fields = next(csvreader0)
    writer1 = csv.writer(out1)
    writer1.writerow(["source_id","ra", "dec", "prob_other", "prob_pms", "prob_be"])         
    for row in csvreader0:   
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('0')
    for row in csvreader1:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('1')
    for row in csvreader2:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('2')
    for row in csvreader3:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('3')
    for row in csvreader4:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('4')
    for row in csvreader5:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('5')
    for row in csvreader6:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('6')
    for row in csvreader7:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('7')
    for row in csvreader8:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('8')
    for row in csvreader9:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('9')
    


with open('Output_Input_sample_v3_good2.csv', 'w') as out1, open('Output_Input_sample_v3_good.csv', 'r') as inp0, open('Output_Input_sample_v310.csv', 'r') as inp10, open('Output_Input_sample_v311.csv', 'r') as inp11, open('Output_Input_sample_v312.csv', 'r') as inp12, open('Output_Input_sample_v313.csv', 'r') as inp13, open('Output_Input_sample_v314.csv', 'r') as inp14, open('Output_Input_sample_v315.csv', 'r') as inp15, open('Output_Input_sample_v316.csv', 'r') as inp16, open('Output_Input_sample_v317.csv', 'r') as inp17, open('Output_Input_sample_v318.csv', 'r') as inp18, open('Output_Input_sample_v319.csv', 'r') as inp19:
    csvreader0 = csv.reader(inp0)
    csvreader10 = csv.reader(inp10)
    csvreader11 = csv.reader(inp11)
    csvreader12 = csv.reader(inp12)
    csvreader13 = csv.reader(inp13)
    csvreader14 = csv.reader(inp14)
    csvreader15 = csv.reader(inp15)
    csvreader16 = csv.reader(inp16)
    csvreader17 = csv.reader(inp17)
    csvreader18 = csv.reader(inp18)
    csvreader19 = csv.reader(inp19)  
    fields = next(csvreader0)
    writer1 = csv.writer(out1)
    writer1.writerow(["source_id","ra", "dec", "prob_other", "prob_pms", "prob_be"])       
    for row in csvreader0:
           writer1.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])
    print('0')    
    for row in csvreader10:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('10')
    for row in csvreader11:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('11')
    for row in csvreader12:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('12')
    for row in csvreader13:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('13')
    for row in csvreader14:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('14')
    for row in csvreader15:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('15')
    for row in csvreader16:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('16')
    for row in csvreader17:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('17')
    for row in csvreader18:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('18')
    for row in csvreader19:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('19')


with open('Output_Input_sample_v3_good3.csv', 'w') as out1, open('Output_Input_sample_v3_good2.csv', 'r') as inp0, open('Output_Input_sample_v320.csv', 'r') as inp20, open('Output_Input_sample_v321.csv', 'r') as inp21, open('Output_Input_sample_v322.csv', 'r') as inp22, open('Output_Input_sample_v323.csv', 'r') as inp23, open('Output_Input_sample_v324.csv', 'r') as inp24, open('Output_Input_sample_v325.csv', 'r') as inp25, open('Output_Input_sample_v326.csv', 'r') as inp26, open('Output_Input_sample_v327.csv', 'r') as inp27, open('Output_Input_sample_v328.csv', 'r') as inp28, open('Output_Input_sample_v329.csv', 'r') as inp29:
    csvreader0 = csv.reader(inp0)
    csvreader20 = csv.reader(inp20)
    csvreader21 = csv.reader(inp21)
    csvreader22 = csv.reader(inp22)
    csvreader23 = csv.reader(inp23)
    csvreader24 = csv.reader(inp24)
    csvreader25 = csv.reader(inp25)
    csvreader26 = csv.reader(inp26)
    csvreader27 = csv.reader(inp27)
    csvreader28 = csv.reader(inp28)
    csvreader29 = csv.reader(inp29)    
    fields = next(csvreader0)
    writer1 = csv.writer(out1)
    writer1.writerow(["source_id","ra", "dec", "prob_other", "prob_pms", "prob_be"])         
    for row in csvreader0:
           writer1.writerow([row[0],row[1],row[2],row[3],row[4],row[5]])
    print('0')
    for row in csvreader20:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('20')
    for row in csvreader21:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('21')
    for row in csvreader22:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('22')
    for row in csvreader23:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('23')
    for row in csvreader24:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('24')
    for row in csvreader25:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('25')
    for row in csvreader26:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('26')
    for row in csvreader27:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('27')
    for row in csvreader28:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('28')
    for row in csvreader29:
           writer1.writerow([row[0],row[1],row[2],row[18],row[19],row[20]])
    print('29')    


df = pd.read_csv('Output_Input_sample_v3_good3.csv', sep=',')
df.sort_values(by=['ra','dec']).to_csv('Output_Input_sample_v3_good_sorted.csv')

    
with open('Output_Input_sample_v3_good_sorted2.csv', 'w') as out1, open('Output_Input_sample_v3_good_sorted.csv', 'r') as inp0:
    csvreader0 = csv.reader(inp0)
    fields = next(csvreader0)
    writer1 = csv.writer(out1)
    writer1.writerow(["source_id","ra", "dec", "prob_other", "prob_pms", "prob_be"])         
    i = 0
    source_id = []
    ra = []
    dec = []
    prob_other = []
    prob_pms = []
    prob_be = [] 
    source_id_mean = []
    ra_mean = []
    dec_mean = []
    prob_other_mean = []      
    prob_pms_mean = []   
    prob_be_mean = []
    prob_other_mean_error = []
    prob_pms_mean_error = []
    prob_be_mean_error = []
    last = 0
    #for row in csvreader0: 
       # row_count = sum(1 for row in csvreader0)
    for row in csvreader0: 
           last = last+1
           source_id.append(float(row[1]))
           ra.append(float(row[2]))
           dec.append(float(row[3]))
           prob_other.append(float(row[4]))
           prob_pms.append(float(row[5]))
           prob_be.append(float(row[6]))
           if i==0:
               print('Row 1')               
               i = 1
           else:               
               if ra[-1] != ra[-2] or dec[-1] != dec[-2]:
                  print('Not Repeated')
                  source_id_mean.append(source_id[-2])
                  ra_mean.append(ra[-2])
                  dec_mean.append(dec[-2])
                  prob_other_mean.append(np.mean(prob_other[:-1]))      
                  prob_pms_mean.append(np.mean(prob_pms[:-1]))   
                  prob_be_mean.append(np.mean(prob_be[:-1]))
                  prob_other_mean_error.append(np.std(prob_other[:-1])/math.sqrt(len(prob_other[:-1])))    
                  prob_pms_mean_error.append(np.std(prob_pms[:-1])/math.sqrt(len(prob_pms[:-1])))  
                  prob_be_mean_error.append(np.std(prob_be[:-1])/math.sqrt(len(prob_be[:-1])))
                  source_id = []                  
                  ra = []
                  dec = []
                  prob_other = []
                  prob_pms = []
                  prob_be = []
                  source_id.append(float(row[1]))
                  ra.append(float(row[2]))
                  dec.append(float(row[3]))
                  prob_other.append(float(row[4]))
                  prob_pms.append(float(row[5]))
                  prob_be.append(float(row[6]))                 
    print('last')
    source_id_mean.append(source_id[-1])
    ra_mean.append(ra[-1])
    dec_mean.append(dec[-1])
    prob_other_mean.append(np.mean(prob_other[:]))      
    prob_pms_mean.append(np.mean(prob_pms[:]))   
    prob_be_mean.append(np.mean(prob_be[:]))
    prob_other_mean_error.append(np.std(prob_other[:])/math.sqrt(len(prob_other[:])))    
    prob_pms_mean_error.append(np.std(prob_pms[:])/math.sqrt(len(prob_pms[:])))  
    prob_be_mean_error.append(np.std(prob_be[:])/math.sqrt(len(prob_be[:])))               
    for a in range(len(ra_mean)):             
        writer1.writerow([source_id_mean[a],ra_mean[a],dec_mean[a],prob_other_mean[a],prob_pms_mean[a],prob_be_mean[a],prob_other_mean_error[a],prob_pms_mean_error[a],prob_be_mean_error[a]])
           
print("--- %s minutes ---" % ((time.time() - start_time)/60))
