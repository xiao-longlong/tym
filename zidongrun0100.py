import time
import datetime
import subprocess

def execute_command(command):
    """执行指定的 Bash 指令"""
    subprocess.run(command, shell=True)

def wait_until(target_time):
    """等待直到达到指定的时间"""
    while True:
        now = datetime.datetime.now()
        if now >= target_time:
            break
        time.sleep(1)  # 每秒检查一次

if __name__ == "__main__":
    # 设置目标时间，例如 2024-09-15 14:30:00
    target_time = datetime.datetime(2024, 9, 16, 1, 1, 1)

    print(f"当前时间：{datetime.datetime.now()}")
    print(f"等待直到指定时间：{target_time}")

    # 等待直到达到目标时间
    wait_until(target_time)

    # 执行 Bash 指令
    command = "python /workspace/tools/train.py -f /workspace/exps/example/custom/GTSRB_yoloxs_ours0100.py -n yolox-s -d 1 -b 8 -expn GTSRB_100210022002300_09160100"
    execute_command(command)

    print("指令执行完毕")