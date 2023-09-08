import javabridge
import bioformats

def start():
    javabridge.start_vm(class_path=bioformats.JARS)
    logback = javabridge.JClassWrapper("loci.common.LogbackTools")
    logback.enableLogging()
    logback.setRootLevel("ERROR")
    print('javabride started')


def stop():
    javabridge.kill_vm()
    print('javabride stopped')
