UNIT filters;

INTERFACE

  PROCEDURE bp_bworth (trptr: pointer; tsa: double; ndat: integer;  flo,fhi: double; ns, zph : integer );


IMPLEMENTATION

  USES Data_Rsc;

  CONST
    MAX_SEC=10;
    M_PI = PI;
{    MAX_SIZE=8192;}
    MAX_SIZE=MaxHistSz;

  TYPE
{    DoubleArray = ARRAY [0..MAX_SIZE] OF double;
    Doubleptr   = ^double;
    DoubleArrayPtr = ^DoubleArray;}

    SingleArray = ARRAY [0..MAX_SIZE-1] OF Single;
    Singleptr   = ^Single;
    SingleArrayPtr = ^SingleArray;

  PROCEDURE bp_bworth (trptr: pointer; tsa: double; ndat: integer;  flo,fhi: double; ns, zph : integer );
      {Butterworth Bandpass filter, translated from C from routine by A. Rietbrock}
      {NOTE - now assumes an input array of singles not doubles}
    VAR
      k, n,m,mm : Integer;
      a,b,c,d,e : ARRAY [0..MAX_SEC+1] OF double;
      f         : ARRAY [0..MAX_SEC+1, 0..5] OF double;
      temp, c1,c2,c3 : double;
      w1,w2,wc,q,p,r,s,cs,x : double;
      tr             : SingleArrayPtr;

    BEGIN
      {Check array limit first, and assign pointer to array}
      IF (ndat > MAX_SIZE) THEN Exit;
      tr := trptr;

      { design filter weights }
      w1:=sin(flo*M_PI*tsa)/cos(flo*M_PI*tsa);
      w2:=sin(fhi*M_PI*tsa)/cos(fhi*M_PI*tsa);
      wc:=w2-w1;
      q:=wc*wc +2.0*w1*w2;
      s:=w1*w1*w2*w2;
      FOR k:=1 TO ns DO
        BEGIN
          c1:=(k+ns);
          c2:=(4*ns);
          c3:=(2.0*c1-1.0)*M_PI/c2;
          cs:=cos(c3);
          p := -2.0*wc*cs;
          r:=p*w1*w2;
          x:=1.0+p+q+r+s;
          a[k]:=wc*wc/x;
          b[k]:=(-4.0 -2.0*p+ 2.0*r+4.0*s)/x;
          c[k]:=(6.0 - 2.0*q +6.0*s)/x;
          d[k]:=(-4.0 +2.0*p -2.0*r +4.0*s)/x;
          e[k]:=(1.0 - p +q-r +s)/x;
        END; {for k}

      { set initial values to 0 }
      FOR n:=0 TO MAX_SEC DO
        FOR m:=0 TO 5 DO
          f[n][m]:=0.0;

      { filtering }
      FOR m:=1 TO ndat DO
        BEGIN
          f[1][5] := tr^[m-1];
          { go thru ns filter sections }
          FOR n:=1 TO ns DO
            BEGIN
              temp:=a[n]*(f[n][5]-2.0*f[n][3] +f[n][1]);
              temp:=temp-b[n]*f[n+1][4]-c[n]*f[n+1][3];
              f[n+1][5]:=temp-d[n]*f[n+1][2]-e[n]*f[n+1][1];
            END;
          { update past values }
          FOR n:=1 TO (ns+1) DO
              FOR mm:=1 TO 4 DO
                  f[n][mm]:=f[n][mm+1];
          { set present data value and continue }
          tr^[m-1] := f[ns+1][5];
        END; {for m}


      IF (zph = 1) THEN
        BEGIN
          { filtering reverse signal }
          FOR m:= ndat DOWNTO 1 DO
            BEGIN
              f[1][5] := tr^[m-1];
              { go thru ns filter sections }
              FOR n:=1 TO ns DO
                BEGIN
                  temp:=a[n]*(f[n][5]-2.0*f[n][3] +f[n][1]);
                  temp:=temp-b[n]*f[n+1][4]-c[n]*f[n+1][3];
                  f[n+1][5]:=temp-d[n]*f[n+1][2]-e[n]*f[n+1][1];
                END;
              { update past values }
              FOR n:=1 TO ns+1 DO
                FOR mm:=1 TO 4 DO
                    f[n][mm]:=f[n][mm+1];
              { set present data value and continue }
              tr^[m-1] := f[ns+1][5];
            END; {for m}
      END; {if zph=1}

    END; {bp_bworth}

BEGIN
END. {filters}
