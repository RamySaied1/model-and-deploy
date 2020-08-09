import sys
import requests

if (len(sys.argv)!=2):
    print("error in parameters")
    exit(1)

url=None
if (sys.argv[1]=='local'):
    url = 'http://localhost:5000/class/'
elif (sys.argv[1]=="global"):
    url = 'http://ramysaied.pythonanywhere.com/class/'
else:
    print("wrong arguments value")
    exit(0)

features_no={ 
"variable1" :"a",
"variable2" :"17,92",
"variable3" :"5,4e-05",
"variable4" :"u",
"variable5" :"g",
"variable6" :"c",
"variable7" :"v",
"variable8" :"1,75",
"variable9" : "f",
"variable10" : "t",
"variable11" : "1",
"variable12" : "t",
"variable13" :"g",
"variable14" : "80",
"variable15" : "5",
"variable17" : "8e+05",
"variable18" : "t",
"variable19" : "0",
} ## expected class no


features_yes={ 
"variable1" :"b",
"variable2" :"33,17",
"variable3" :"0,000104",
"variable4" :"u",
"variable5" :"g",
"variable6" :"r",
"variable7" :"h",
"variable8" :"6,5",
"variable9" : "t",
"variable10" : "f",
"variable11" : "0",
"variable12" : "t",
"variable13" :"g",
"variable14" : "164",
"variable15" : "31285",
"variable17" : "1640000",
"variable18" : "f",
"variable19" : "1",
} ## expected class yes
r = requests.post(url,json=features_no)
print(r)
print(r.json())

r = requests.post(url,json=features_yes)
print(r)
print(r.json())