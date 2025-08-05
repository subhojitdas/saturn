import time

def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)


import asyncio

async def cpu_task(n):
    return fib(n)

async def main():
    start = time.time()
    await asyncio.gather(cpu_task(35), cpu_task(35))
    end = time.time()
    print("time taken:", end-start)


import threading

# def worker(n):
#     print(fib(n))



# threads = [
#     threading.Thread(target=worker, args=(35,)),
#     threading.Thread(target=worker, args=(35,))
# ]
#
# start = time.time()
# for t in threads: t.start()
# for t in threads: t.join()
# end = time.time()
# print("time taken:", end-start)

# asyncio.run(main())


from multiprocessing import Process

def worker(n):
    print(fib(n))

if __name__ == "__main__":
    processes = [
        Process(target=worker, args=(100,)),
        Process(target=worker, args=(100,))
    ]
    start = time.time()
    for p in processes: p.start()
    for p in processes: p.join()
    end = time.time()
    print("time taken:", end-start)