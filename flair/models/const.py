class _const:
    """
    自定义常量：（1）命名全部大写；（2）值不可修改
    """
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        #if self.__dict__.haskey(name):
        if name in self.__dict__.keys():
            raise self.ConstError('Can not change const.{0}'.format(name))
        if not name.isupper():
            raise self.ConstCaseError(
                'const name {0} is not all uppercase.'.format(name))
        self.__dict__[name] = value


import sys
sys.modules[__name__] = _const()