#-*- coding=utf8 -*-
from libDL import *
      
if __name__ == '__main__':
    #read data from file
    # file_name = sys.argv[1]
    # train_name = sys.argv[2]
    file_name = 'learn_data'
    train_name = 'train_data'
    file_name = '/Users/rex/z_thesis/CNN_test/a8.txt'
    # train_name = 'a1.txt'
    input_node = Open_data(file_name)

    #read training data
    d = Open_data(train_name)

    #Prepare weight
    w = []

    #decide kernel size
    conv_kernel = 4
    pool_kernel = 2

    #Conv weight
    w.append(MakeWeight(4,1))
    #M_Pool weight
    next_nodelen=(len(list(input_node[0]))-conv_kernel+1)/pool_kernel+1
    #FC weight(node_length and output node num)
    w.append(MakeWeight(next_nodelen, 3))
    
    #Prepare layer object
    Input = Layer(0)
    Conv_Layer1 = Layer(conv_kernel)
    Pool_Layer1 = Layer(pool_kernel)
    Out_Layer1 = Layer(0)


    """
    ------------------------------
            start learning
    ------------------------------
    """
    count = 0

    for z in range(len(input_node)):

        Input.node = input_node[z]
        Pass_Conv(Input, Conv_Layer1, w[0][0])
        #Pooling must remember pre node
        Pool_Layer1.bp_node = Conv_Layer1.node
        Pass_Max_Pool(Conv_Layer1, Pool_Layer1, pool_kernel)
        Pass_FC_Out(Pool_Layer1, Out_Layer1, w[1])
        
        #print output node
        print("Num : %d Output = [" % z,  end="")
        for i in Out_Layer1.node:
            print(" %f " % i, end="")
        print (']')

        
        """
        ------------------------------
          start evaluating accuracy
        ------------------------------
        """

        
        max = [0.0, 0]
        for i in range(len(Out_Layer1.node)):
            if Out_Layer1.node[i] > max[0]:
                max[0] = Out_Layer1.node[i]
                max[1] = i
        if d[z][max[1]] == 1:
            count += 1


        """
        ------------------------------
            start back propagation
        ------------------------------
        """


        #Calc delta
        delta_3 = Cross_Entropy(Out_Layer1.node, d[z])
        delta_2 = Out_FC_Delta(Pool_Layer1.node, delta_3, w[1])
        delta_1 = Max_Pool_Delta(Pool_Layer1, delta_2)
        delta_0 = Conv_Delta(Conv_Layer1, delta_1, w[0][0])
        #Update weight
        w[1] = FC_Update(Pool_Layer1.node, delta_3, w[1])
        w[0][0] = Conv_Update(Input.node, delta_1, w[0][0])


    #accuracy
    print("\n%.2f%% correct." % (float(count) / float(len(input_node)) * 100.0))

pass
    
