import requests
import time
from new_agent.app.app import run_server
import os 
from shutil import copyfile


def init():
    base_url = "http://127.0.0.1:8080"
    requests.post(base_url+"/usr/usr_login",json={"usr_name":"usr1","password":"123"})
    requests.post(base_url+"/db/new_collection",json={"collection_name":"Collection1"})
    file_list = os.listdir("document/")
    for file_name in file_list:
        copyfile("document/"+file_name, "cache/"+file_name)
        requests.post(base_url+"/db/add_document",json={"collection_name":"Collection1","file_name":file_name,"description":"document1"})
    response = requests.post(base_url+"/db/get_collection",json={})
    print(response.json())

    response = requests.post(base_url+"/db/get_document",json={"collection_name":"Collection1"})
    print(response.json())

def example():
    base_url = "http://127.0.0.1:8080"
    response = requests.post(base_url+"/usr/usr_login",json={"usr_name":"usr1","password":"123"})
    print(response.json())

    response = requests.post(base_url+"/db/new_collection",json={"collection_name":"Collection2"})
    print(response.json())

    file_name = "RankEncoder.pdf"
    copyfile("document/"+file_name, "cache/"+file_name)
    response = requests.post(base_url+"/db/add_document",json={"collection_name":"Collection2","file_name":file_name,"description":"document1"})
    print(response.json())

    response = requests.post(base_url+"/db/get_collection",json={})
    print(response.json())

    response = requests.post(base_url+"/db/get_document",json={"collection_name":"Collection2"})
    print(response.json())

    response = requests.post(base_url+"/db/delete_document",json={"collection_name":"Collection2","file_name":"RankEncoder.pdf"})
    print(response.json())

    response = requests.post(base_url+"/db/delete_collection",json={"collection_name":"Collection2"})
    print(response.json())

def main():
    run_server()

    time.sleep(1)

    example()
    #init()

if __name__ == "__main__":
    main()