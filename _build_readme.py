res = ""
with open("README_template.rst", "r") as f:
    for line in f.readlines():
        line = line.strip("\n").replace(".. include:: ", "").replace(".. literalinclude:: ../LICENSE", "").replace("<../LICENSE>", "<LICENSE>")
        if line.endswith(".rst"):
            tf = open(line, "r").read()
            res += tf
        else:
            res += line + "\n"

    print(res)

with open("README.rst", "w+") as f:
    f.write(res)
