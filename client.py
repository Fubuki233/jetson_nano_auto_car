import socket
from pynput import keyboard
# 1.创建socket
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2. 链接服务器
server_addr = ("192.168.232.206", 7788)
tcp_socket.connect(server_addr)
def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        #print(format(key.char))
        if format(key.char)=='w':
            print('running in straight direction')
            send_data ='w'
            tcp_socket.send(send_data.encode("gbk"))
            #car.Car_Run(150, 150)
            #time.sleep(0.1)
            #car.Car_Stop()
        if format(key.char)=='s':
            print('running in straight direction')
            send_data ='s'
            tcp_socket.send(send_data.encode("gbk"))
            #car.Car_Run(150, 150)
            #time.sleep(0.1)
            #car.Car_Stop()

    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    send_data = '-'
    print(send_data)
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()
# 3. 发送数据
while(1):
    send_data = input("请输入要发送的数据：")
    tcp_socket.send(send_data.encode("gbk"))

# 4. 关闭套接字
tcp_socket.close()