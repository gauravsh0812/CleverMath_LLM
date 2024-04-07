def test_categorized_accuracy():
    labels = open("logs/labels.txt").readlines()[1:]

    totaladd,totalsub,totalmul,totaladv=0,0,0,0
    add,sub,adv,mul,count=0,0,0,0,0

    for _l in labels:
    
        _l = _l.replace("\n","").strip()
        cat,truth, pred = _l.split()    
        if cat == "addition":
            totaladd+=1
            if truth == pred:
                add+=1
                count+=1
        
        elif cat == "subtraction":
            totalsub +=1
            if truth == pred:
                sub+=1
                count+=1

        elif cat == "adversarial":
            totaladv +=1
            if truth == pred:
                adv+=1
                count+=1

        elif cat == "subtraction-multihop":
            totalmul+=1
            if truth == pred:
                mul+=1
                count+=1

    print("Accuracies: ")
    print("Total Accuracy: ", count/len(labels))
    print("Addition Accuracy: ", add/totaladd)
    print("Subtraction Accuracy: ", sub/totalsub)
    print("Adversarial Accuracy: ", adv/totaladv)
    print("Sub-Multihop Accuracy: ", mul/totalmul)