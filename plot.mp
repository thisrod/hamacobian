% See plot.py

path marker;
marker = subpath ((3.2/7, 5.37/7)*(length fullcircle)) of fullcircle
	-- (0,0)
	-- subpath ((5.63/7, 7.8/7)*(length fullcircle)) of fullcircle
	-- point (7.8/7*length fullcircle) of fullcircle
	-- 1.1*(point (9/7 length fullcircle) of fullcircle)
	--point (10.2/7*length fullcircle) of fullcircle;

def mark(expr ax, ay, w, f) =
	draw marker scaled (w*1cm) rotated f shifted ((ax, ay)*1in);
enddef;