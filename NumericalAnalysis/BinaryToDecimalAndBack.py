import math


def binaryToDecimal(number):
    returnVal = 0.0
    returnDec = 0.0
    try:
        decimalExist = str(number).index(".")
    except ValueError:
        decimalExist = -1
    nonDecimalVal = int(number)
    decimalVal = 0.0
    if decimalExist != -1:
        decimalVal = "0." + str(number)[decimalExist + 1 :]
    i = 0
    lengthDec = len(str(decimalVal)) - 2
    while nonDecimalVal >= 10:
        returnVal += int((nonDecimalVal % 10)) * math.pow(2, i)
        nonDecimalVal /= 10
        i += 1
    returnVal += int((nonDecimalVal % 10)) * math.pow(2, i)
    b = -1
    count = 1
    decimalVal = float(decimalVal)

    while lengthDec != 0:
        returnDec += int((decimalVal * math.pow(10, count))) % 10 * math.pow(2, -count)
        count += 1
        lengthDec -= 1

    return returnVal + returnDec


def DecimalToBinary(number):
    returnString = "0."
    count = 0
    while number % 1 != 0 and count < 50:
        returnString = returnString + str(int(number * 2))
        number = number * 2 % 1
        count += 1

    if count >= 50:
        returnString += "..."
    return returnString


print(binaryToDecimal(110110.111))
print(binaryToDecimal(1110.1010))
print(binaryToDecimal(111000.11011))
print(binaryToDecimal(1.1))
print(binaryToDecimal(1000001.100001))

print(DecimalToBinary(0.678))
print(DecimalToBinary(0.765863))
print(DecimalToBinary(0.11111112))
print(DecimalToBinary(0.4432))
print(DecimalToBinary(0.998))
