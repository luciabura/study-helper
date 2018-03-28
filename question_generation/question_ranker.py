import javaobj


def read_file(filepath, stream=False):
    if stream:
        return open(filepath, "rb")
    else:
        with open(filepath, 'rb') as filep:
            return filep.read()


marshall = javaobj.JavaObjectUnmarshaller(read_file('/home/lucia/part-II/study-helper/text_processing/linear-regression-ranker-reg500.ser', stream=True))
pobj = marshall.readObject()
# jobj_read = read_file('/home/lucia/part-II/study-helper/text_processing/linear-regression-ranker-reg500.ser')
# pobj = javaobj.loads(jobj_read)
pobj.name

print(pobj.name)

