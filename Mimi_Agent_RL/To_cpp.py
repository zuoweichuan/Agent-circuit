def V2c (Vpath,Cpath):
    with open(Vpath,'r') as v:
        with open(Cpath,'w') as cpp:
            cpp.write('#include <stdio.h>\n')
            cpp.write('#include <stdbool.h>\n')
            cpp.write('#include <math.h>\n\n')
            cpp.write('int main(){\n')
            cpp.write('    bool A[8];\n')
            cpp.write('    bool B[8];\n')
            cpp.write('    bool O[16];\n')
            cpp.write('    bool N[3000];\n\n')
            cpp.write('    int result = 0;\n')
            cpp.write('    int real = 0;\n')
            cpp.write('    unsigned long long cont = 0;\n')
            cpp.write('    unsigned long long error = 0;\n')
            cpp.write('    float snr = 0;\n')
            cpp.write('    unsigned long long allreal2 = 0;\n\n')
            cpp.write('    for(int mc = 0; mc < 100000 ; mc++){\n')
            cpp.write('        int i = rand() % 256-128;\n')
            cpp.write('        int j = rand() % 256-128;\n')
            cpp.write('            for(int k = 0; k < 8; k++){\n')
            cpp.write('                A[k] = (i >> k) & 1;\n')
            cpp.write('                B[k] = (j >> k) & 1;\n')
            cpp.write('            }\n')
            for line in v:
                if line.strip().startswith('assign'):
                    line = line.replace('assign', '').strip()
                    line = line.replace(']N', ']').strip()
                    line = line.replace('1\'b', '').strip()
                    cpp.write('            ' + line + '\n')
            cpp.write('            for(int k = 0; k < 15; k++){\n')
            cpp.write('                result += O[k]* pow(2,k);\n')
            cpp.write('            }\n')
            cpp.write('            result -= O[15]*pow(2,15);\n')
            cpp.write('            real = i*j;\n')
            cpp.write('            error += pow(real-result,2);\n')
            cpp.write('            allreal2 += pow(real,2);\n')
            cpp.write('            cont++;\n')
            cpp.write('            result = 0;\n')
            cpp.write('    }\n')
            cpp.write('    snr = 10*log10(allreal2/error);\n')
            cpp.write('    printf("//MSE = %llu\\n",error/cont);\n')
            cpp.write('    printf("//SNR = %f",snr);\n')
            cpp.write('    return 0;\n')
            cpp.write('}\n')   

import subprocess
import re

def run_cpp_file(file_path):
    # 编译.cpp文件
    compile_process = subprocess.run(['g++', file_path, '-o', r'output.exe'], capture_output=True, text=True)
    
    if compile_process.returncode != 0:
        print("编译错误：")
        print(compile_process.stderr)
        run_process = '//MSE = 1000000\n'
        return run_process
    
    # 运行编译后的可执行文件
    run_process = subprocess.run([r'output.exe'], capture_output=True, text=True)
    
    if run_process.returncode != 0:
        print("运行错误：")
        print(run_process.stderr)
        return
    
    return run_process.stdout

if __name__ == '__main__':
    V2c(r'temp.v',r'multi.cpp')
    output = run_cpp_file(r'multi.cpp')
    print(output)