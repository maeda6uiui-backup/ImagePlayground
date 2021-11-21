if __name__=="__main__":
    in_size=int(input("in_size="))
    kernel_size=int(input("kernel_size="))
    stride=int(input("stride="))
    padding=int(input("padding="))

    out_size=(in_size-1)*stride-2*padding+kernel_size
    print(out_size)
