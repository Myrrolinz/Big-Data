RESULTS_FOLDER = "../Results/"

result_LFM_file = RESULTS_FOLDER + "result_LFM.txt"
result_Regression_file = RESULTS_FOLDER + "result_Regression.txt"
result_CF_user_file = RESULTS_FOLDER + "result_CF_user.txt"
final_result_file = RESULTS_FOLDER + "result.txt"


def read_result_file(file):
    result = dict()

    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            # 对于文件中的每一行
            if len(line.split("|")) == 2:
                # 表示此时这一行是userID|num
                curr_user = eval(line.split("|")[0])
                curr_num = eval(line.split("|")[1])
                result[curr_user] = dict()
            else:
                # 表示此时这一行是itemID score
                # 判断当前这一行对不对
                if len(result[curr_user]) >= curr_num:
                    print("Data Error: Form of current line is 'itemID score', but is should be 'userID|num'")
                    exit(1)

                itemID = eval(line.split()[0])
                score = eval(line.split()[1])
                result[curr_user][itemID] = score

    return result


def save_result(file_name, data):
    print("Saving result file...")

    with open(file_name, "w") as f:
        for user in data.keys():
            f.write(str(user) + "|" + str(len(data[user].keys())) + "\n")
            for item in data[user].keys():
                f.write(str(item) + " " + str(data[user][item]) + "\n")

    print(f"Test result file in save as {file_name}.")


def test():
    temp_result = 53.4
    if temp_result % 10 >= 5:
        temp_result = (int(temp_result / 10) + 1) * 10
    else:
        temp_result = int(temp_result / 10) * 10

    print(temp_result)


if __name__ == "__main__":
    LFM_result = read_result_file(result_LFM_file)
    CF_result = read_result_file(result_CF_user_file)
    Reg_result = read_result_file(result_Regression_file)

    result = dict()

    for user in CF_result.keys():
        result[user] = dict()
        for item in CF_result[user].keys():
            temp_result = 0.2*CF_result[user][item] + 0.4*LFM_result[user][item] + 0.4*Reg_result[user][item]
            if temp_result % 10 >= 5:
                temp_result = (int(temp_result / 10) + 1) * 10
            else:
                temp_result = int(temp_result / 10) * 10
            result[user][item] = temp_result

    save_result(final_result_file, result)
    # test()
