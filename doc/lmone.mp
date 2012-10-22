% What problem Levenberg-Marquardt solves, in one dimension

u=1cm;	% Master scale

% Elements of the problem

pair spot;  spot = (0,1.9)*u;

path mfld;  pair w[];
w1= (-1.5,-2)*u;  w2=(0,0);  w3=w1 xscaled -1;
mfld = w1{2/3[w2,spot]-w1}...w2...{w3-2/3[w2,spot]}w3;


beginfig(1)
% Newton-naive approach

dotlabel.top("X", spot);

save s,m,t,mpt,c,tgt;  pair m,mpt[],tgt;
t := 0.2;
mpt1 := point t of mfld;

for i = 1 upto 2:
	m := direction t of mfld;
	z[i+1] = mpt[i] + whatever*m = spot + whatever*(m rotated 90);
	tgt := z[i+1]-mpt[i];
	draw mpt[i]--(mpt[i]+tgt+(0.6+0.05 normaldeviate)*u*unitvector tgt) withcolor 0.5 white;
	s := arclength(subpath (0, t) of mfld)
		if m dotprod tgt > 0: + else: - fi abs tgt;
	t := arctime s of mfld;
	mpt[i+1] := point t of mfld;
endfor

draw mfld withpen (currentpen scaled 1.5);
dotlabel.lrt("A", mpt1);
dotlabel.rt("b", z2);
dotlabel.lrt("B", mpt2);
dotlabel.rt("c", z3);
dotlabel.lrt("C", mpt3);

endfig;


beginfig(2)
% Levenberg-Marquardt, with limits

dotlabel.top("X", spot);

save s,m,t,mpt,c,tgt,tls;  pair m,mpt[],tgt;
t := 0.2;
mpt1 := point t of mfld;
tls1 = 1.6u;
tls2 = 0.5u;
tls3 = 2u;

for i = 1 upto 2:
	m := direction t of mfld;
	z[2i+1] = mpt[i] + whatever*m = spot + whatever*(m rotated 90);
	tgt := tls[i]*unitvector(z[2i+1]-mpt[i]);
	z[2i] = mpt[i]+tgt;
	draw mpt[i]--(z[2i+1]+(0.6+0.05 normaldeviate)*u*unitvector tgt) withcolor 0.5 white;
	s := arclength(subpath (0, t) of mfld)
		if m dotprod tgt > 0: + else: - fi abs tgt;
	t := arctime s of mfld;
	mpt[i+1] := point t of mfld;
endfor

draw mfld withpen (currentpen scaled 1.5);
dotlabel.lrt("A", mpt1);
dotlabel.ulft("b", z2);
dotlabel.lrt("B", mpt2);
dotlabel.rt("c", z4);
dotlabel.bot("C", mpt3);

endfig;

end