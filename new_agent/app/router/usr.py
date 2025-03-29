from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..args import db,global_memory


router = APIRouter()

class LoginRequest(BaseModel):
    usr_name: str
    password: str

@router.post("/usr_login")
async def web_usr_login(request: LoginRequest):
    return usr_login(request.usr_name, request.password)

@router.post("/usr_register")
async def web_usr_register(request: LoginRequest):
    return usr_register(request.usr_name, request.password)

def usr_register(usr_name: str, password: str):
    if db.usr_dict.get(usr_name, -1) != -1:
        return {"status": "该用户名已存在，换一个吧", "code": -1}
    else:
        db.usr_dict[usr_name] = password
        return {"status": "注册成功", "code": 1}

def usr_login(usr_name: str, password: str):
    if db.usr_dict.get(usr_name, -1) == -1:
        return {"status": "用户不存在", "code": -1}
    elif db.usr_dict[usr_name] != password:
        return {"status": "密码错误", "code": -2}
    else:
        load_conversation(usr_name)
        return {"status": "登录成功，已成功加载历史对话", "code": 1}

async def load_conversation(usr_name: str):
    """从文件加载对话历史"""
    file_path = f"./chat/{usr_name}.json"
    global_memory.load_memory_from_file(file_path)
    print(f"已从 {file_path} 加载对话历史")