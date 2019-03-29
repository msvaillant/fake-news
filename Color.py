import sys

colours = {
	"default"    :    "\033[0m",
	# style
	"bold"       :    "\033[1m",
	"underline"  :    "\033[4m",
	"blink"      :    "\033[5m",
	"reverse"    :    "\033[7m",
	"concealed"  :    "\033[8m",
	# couleur texte
	"black"      :    "\033[30m",
	"red"        :    "\033[31m",
	"green"      :    "\033[32m",
	"yellow"     :    "\033[33m",
	"blue"       :    "\033[34m",
	"magenta"    :    "\033[35m",
	"cyan"       :    "\033[36m",
	"white"      :    "\033[37m",
	# couleur fond
	"on_black"   :    "\033[40m",
	"on_red"     :    "\033[41m",
	"on_green"   :    "\033[42m",
	"on_yellow"  :    "\033[43m",
	"on_blue"    :    "\033[44m",
	"on_magenta" :    "\033[45m",
	"on_cyan"    :    "\033[46m",
	"on_white"   :    "\033[47m" }

def color(nom):
	"""
	Change the color of the terminal output
	"""
	# on évite print à cause du '\n' inséré automatiquement
	sys.stdout.write(colours[nom])

def display(str,c='default'):
	"""
	Pretty print for the terminal, default color is the defaut color of the terminal
	"""
	color(c)
	print(str)
	color('default')
def error(str):
	"""
	Show a pretty error and quit the app
	"""
	color('red')
	print("============")
	print(str)
	print("============")
	color('default')
	exit()
