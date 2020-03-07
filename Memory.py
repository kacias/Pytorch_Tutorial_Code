#=========================
#파이썬 메모리 관리
#https://github.com/zhuyifei1999/guppy3
#pip install guppy

from guppy import hpy
h = hpy()
print (h.heap())



##########################################################################################
class HeapMon:
	#=====================================================================================
	def __init__(self):
		try:
			from guppy import hpy
			self.enabled = True
		except:
			self.enabled = False
		if self.enabled:
			self._h = hpy()
		self.hsize = 0
		self.hdiff = 0

	#=====================================================================================
	@staticmethod
	def getReadableSize(lv):
		if not isinstance(lv, (int, int)):
			return '0'
		if lv >= 1024*1024*1024*1024:
			s = "%4.2f TB" % (float(lv)/(1024*1024*1024*1024))
		elif lv >= 1024*1024*1024:
			s = "%4.2f GB" % (float(lv)/(1024*1024*1024))
		elif lv >= 1024*1024:
			s = "%4.2f MB" % (float(lv)/(1024*1024))
		elif lv >= 1024:
			s = "%4.2f KB" % (float(lv)/1024)
		else:
			s = "%d B" % lv
		return s
	#=====================================================================================
	def __repr__(self):
		if not self.enabled:
			return 'Not enabled. guppy module not found!'
		if self.hdiff > 0:
			s = 'Total %s, %s incresed' % \
			    (self.getReadableSize(self.hsize), self.getReadableSize(self.hdiff))
		elif self.hdiff < 0:
			s = 'Total %s, %s decresed' % \
			    (self.getReadableSize(self.hsize), self.getReadableSize(-self.hdiff))
		else:
			s = 'Total %s, not changed' % self.getReadableSize(self.hsize)
		return s
	#=====================================================================================
	def getHeap(self):
		if not self.enabled:
			return None
		return str(self._h.heap())
	#=====================================================================================
	def check(self, msg=''):
		if not self.enabled:
			return 'Not enabled. guppy module not found!'
		hdr = self.getHeap().split('\n')[0]
		chsize = int(hdr.split()[-2])
		self.hdiff = chsize - self.hsize
		self.hsize = chsize
		return '%s: %s'% (msg, str(self))




##########################################################################################
hm = HeapMon() # 여기에 글로벌로 hm 이라는 메모리 힙 모니터링을 위한 인스턴스를 만들어 놓았습니다.

##########################################################################################
def do_main():
	print (hm.check('start do_main')) # 함수를 들어와서 메모리 가감을 찍어봅니다
	biglist = []
	for i in range(10000):
		biglist.append(i)		# 백만개의 정수를 담고있는 list를 생성해봅니다
	print (hm.check('end do_main')) # 함수를 종료하기 전에 메모리 가감을 찍어봅니다

##########################################################################################
if __name__ == '__main__':
	print (hm.check('before do_main')) # 메인에서 do_main() 함수 호출전에 최초 메모리 가감을 찍어봅니다
	do_main()
	print (hm.check('after do_main')) # do_main() 함수 호출 후에 메모리 가감을 찍어봅니다
