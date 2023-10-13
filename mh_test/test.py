import yaml

with open("test.yaml") as f:
    
    x = yaml.load(f, Loader=yaml.FullLoader)
print(x)
print(type(x))