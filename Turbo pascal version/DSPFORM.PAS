UNIT DspForm;

  INTERFACE

     USES Funct, Data_Rsc, Fourier, Setts, Filters;

     VAR
       Hnum0 : Integer;
       Hmap : HistMapRec;
       GRs  : GRrec;

     PROCEDURE Parse (Hst : HistPtr;  Pstring : String);


  IMPLEMENTATION


     PROCEDURE Parse (Hst : HistPtr;  Pstring : String);
         {Hmap:HstMapRec & GRs:GRrec are assumed to exist externally.
                            Not passed as parameters due to recursion}
       CONST
         MaxParams = 5;
       VAR
         ParamCount : Integer;
         Params     : ARRAY [1..MaxParams] OF Single;

       PROCEDURE ParseBrackets (bHst: HistPtr);
         VAR
           Count, Pos : Word;
         BEGIN
           Count := 1;
           Pos   := 1;
           REPEAT
             Inc (Pos);
             IF (Pstring[Pos] = '(' )
                THEN Inc (Count);
             IF (Pstring[Pos] = ')' )
                THEN Dec (Count);
           UNTIL (Count = 0);
           Parse  (bHst, Copy (Pstring, 2, (Pos-2) )  );
           Delete (Pstring, 1, Pos);
         END; {ParseBrackets}

       PROCEDURE ParseParam;
         VAR
           Pos, Err, Hnum, i  : Word;
         BEGIN
           ParamCount := 0;
           WHILE (Pstring[1] = ' ') DO
             Delete (Pstring, 1, 1);
           WHILE ( (Pstring<>'') AND (Pstring[1] = ',') ) DO
             BEGIN
               Inc (ParamCount);
               Pos   := 2;
               WHILE Pstring [Pos] IN ['0'..'9','.','-','+','e','E'] DO
                 Inc (Pos);
               Val ( Copy(Pstring,2,(Pos-2)), Params[ParamCount], Err);
               Delete (Pstring, 1, Pos-1);
               WHILE (Pstring[1] = ' ') DO
                 Delete (Pstring, 1, 1);
             END;
           FOR i:=ParamCount+1 TO MaxParams DO
             Params[i] := 0;
         END; {ParseParams}


       PROCEDURE ParseHnum;
         VAR
           Pos, Err, Hnum  : Word;
         BEGIN
           Pos   := 2;
           WHILE Pstring [Pos] IN ['0'..'9'] DO
             Inc (Pos);
           Val ( Copy(Pstring,2,(Pos-2)), Hnum, Err);
           DI_ReadHist ( Hnum, Hst);
           Hmap := HistMap^[Hnum];
           IF (Hmap.Ax1 IN ['T','t'])
              THEN GRs.xstr := 's'
              ELSE GRs.xstr := 'm';
           GRs.ScalT := LinScal;
           Delete (Pstring, 1, Pos);
         END; {ParseHnum}

       PROCEDURE Integrate;
         VAR
           t            : Word;
           Prev1, Prev2 : Single;
         BEGIN
           Prev2   := 2 * Hst^[1] - Hst^[2];
           FOR t := 1 TO Hmap.Tqty  DO
             BEGIN
               Prev1   := Hst^[t];
               Hst^[t] := (Prev1 + Prev2) * 0.5 * Hmap.Dt;
{               Hst^[t] := Hst^[t-1] + (Prev1 + Prev2) / 2 * Hmap.Dt;}
               Prev2   := Prev1;
             END;
         END; {Integrate}

       PROCEDURE Differentiate;
         VAR
           t            : Word;
           Prev1, Prev2 : Single;
         BEGIN
           Prev2 := 2 * Hst^[Hmap.Tqty] - Hst^[Hmap.Tqty-1];
           FOR t := Hmap.Tqty DOWNTO 1 DO
             BEGIN
               Prev1   := Hst^[t];
               Hst^[t] := (Prev2 - Prev1) / Hmap.Dt;
               Prev2   := Prev1;
             END;
         END; {Differentiate}

(*       PROCEDURE Differentiate;
         VAR
           t            : Word;
           Prev1, Prev2 : Single;
         BEGIN
           Prev2   := 2 * Hst^[1] - Hst^[2];
           FOR t := 1 TO Hmap.Tqty  DO
             BEGIN
               Prev1   := Hst^[t];
               Hst^[t] := (Prev1 - Prev2) / Hmap.Dt;
               Hst^[t] := (Prev1 - Prev2) / Hmap.Dt;
{               Hst^[t] := 2 * (Prev1 - Prev2) / Hmap.Dt - Hst^[t-1];}
               Prev2   := Prev1;
             END;
         END; {Differentiate} *)

       PROCEDURE SetTaperParams (ParType, ParStart: Word;
                                 VAR ts1,ts2,tn: Word);
         VAR
           p : Word;
         BEGIN
           p:=ParStart;
           ts1 := TRUNC(0.01*Sets.LowTaper*Hmap.Tqty);
           ts2 := TRUNC(0.01*Sets.HighTaper*Hmap.Tqty);
           tn  := TRUNC(0.01*Sets.MaxTaper*Hmap.Tqty);
           {Percentages}
           IF (ParType = 0) THEN BEGIN
                IF (ParamCount >= p)   THEN ts1 := TRUNC(0.01*Params[p]*(Hmap.Tqty-1)+1);
                IF (ParamCount >= p+1) THEN ts2 := TRUNC(0.01*Params[p+1]*(Hmap.Tqty-1)+1);
                IF (ParamCount >= p+2) THEN tn  := TRUNC(0.01*Params[p+2]*Hmap.Tqty);
              END;
           {Absolute - no checks!}
           IF (ParType = 1) THEN BEGIN
                IF (ParamCount >= p)   THEN ts1 := TRUNC((Params[p]-Hmap.Tstart)/Hmap.Dt+1);
                IF (ParamCount >= p+1) THEN ts2 := TRUNC((Params[p+1]-Hmap.Tstart)/Hmap.Dt+1);
                IF (ParamCount >= p+2) THEN tn  := TRUNC(Params[p+2]/Hmap.Dt);
              END;
         END; {SetTaperParams}

       PROCEDURE Extract;
           {Extract part of waveform - apply exponential Taper on each side}
         VAR
           t,tt,tn,ts1,ts2,nt : Word;
           ParType            : Word;  {0: percentages; 1: absolute times}
           ParStart           : Word;
           ts0,tsf            : Integer;
           tau                : Single;
         BEGIN
           ParType:=0;
           IF (ParamCount >= 1)
              THEN IF (TRUNC(Params[1])=1) THEN ParType:=1;
           ParStart:=2;
           SetTaperParams(ParType,ParStart,ts1,ts2,tn);

           IF ((ts1<1) OR (ts1>Hmap.Tqty)) THEN ts1:=1;
           IF ((ts2<ts1) OR (ts2>Hmap.Tqty)) THEN ts2:=Hmap.Tqty;

           ts0:=ts1-tn; IF (ts0<1) THEN ts0:=1;
           tsf:=ts2+tn; IF (tsf>Hmap.Tqty) THEN tsf:=Hmap.Tqty;

           IF (ParamCount >= 5) THEN nt := TRUNC(Params[4]) ELSE nt:=1;
           IF ((nt<1) OR (nt>100)) THEN nt:=1;

           FOR tt:=1 TO (tsf-ts0+1) DO
             IF (tt MOD nt = 0) THEN
                BEGIN
                  tau:=0; t:=tt+ts0-1;
                  IF (t<ts1) THEN tau := (ts1-t)/tn;
                  IF (t>ts2) THEN tau := (t-ts2)/tn;
                  Hst^[tt DIV nt] := Hst^[t] * exp(-10*tau*tau);
                END;
           WITH Hmap DO
             BEGIN
               Tstart := Tstart + (ts0-1)*dt;
               Time   := Tstart + (tsf-ts0)*dt;
               Tqty := ((tsf-ts0) DIV nt) + 1;
               dt := dt*nt;
             END; {With}
         END; {Extract}

       PROCEDURE SetFiltParams (ParType, ParStart: Word;
                                 VAR f1,f2: Single);
         VAR
           p       : Word;
           MaxFreq : Single;
         BEGIN
           {f1,f2 - mostly interpreted as a stopband; one case as a passband}
           p:=ParStart;
           MaxFreq := 0.5*(1/Hmap.Dt);
           f1 := Sets.LowFreq  / 100;
           f2 := Sets.HighFreq / 100;

           {Percentages}
           IF (ParType = 0) THEN BEGIN
                IF (ParamCount >= p)   THEN f1 := Params[p] / 100;
                IF (ParamCount >= p+1) THEN f2 := Params[p+1] / 100;
              END;

           {Absolute - no checks!}
           IF (ParType = 1) THEN BEGIN
                IF (ParamCount >= p)   THEN f1 := Params[p] / MaxFreq;
                IF (ParamCount >= p+1) THEN f2 := Params[p+1] / MaxFreq;
              END;
         END; {SetFiltParams}

       PROCEDURE ApplyQ (Q, C, L : single);
           {Apply Q, given Q, C-wavespeed, L-propagation distance}
         VAR
           N,i,st,fin          : Integer;
           Four                : FourArray;
           f,df,Qterm,Qfact    : Single;
         BEGIN
           WITH Hmap DO
             BEGIN
               N := TRUNC (Ln(Tqty-1) / Ln(2)) + 1;
               N := N + Sets.FreqWind;
               N := ROUND ( EXP(N*Ln(2)) );
               IF (N > F_MaxEl) THEN N := F_MaxEl;
               st  := (N - Tqty) DIV 2;
               fin := st + Tqty - 1;

               {Zero Padding}
               FOR i := 0 TO N-1 DO
                 Four[i] := Compl0;
               FOR i := 1 TO (Tqty) DO
                 Four[(st-1)+i].Rl := Hst^[i];

               FFT (Four, N, F_Trform);

               {Apply Q}
               df := 1/dT/N; f:=0;
               Qterm := -(2*pi*L)/(2*C*Q);
               FOR i := 0 TO ((N DIV 2)-1) DO
                 BEGIN
                   Qfact:=exp(Qterm*f);
                   Four[i].Rl := Four[i].Rl*Qfact;
                   Four[i].Im := Four[i].Im*Qfact;
                   Four[N-1-i].Rl := Four[N-1-i].Rl*Qfact;
                   Four[N-1-i].Im := Four[N-1-i].Im*Qfact;
                   f := f+df;
                 END;

               FFT (Four, N, F_Inverse);
               {Tqty := N; Tstart:=Tstart-(st*dT); Time:=Tstart+((Tqty-1)*dT);}
               FOR i := 1 TO Tqty DO
                 Hst^[i] := Four[(st-1)+i].Rl;

             END; {With Hmap}
         END; {ApplyQ}

       PROCEDURE Fourier (Inverse : Boolean;  Filt : F_Filt;  Repr : F_Repr);
         VAR
           N,i,st,fin          : Integer;
           LowFreq, HighFreq   : Single;
           LowTaper, HighTaper,
             MaxTaper          : Single;
           Four                : FourArray;
           Sign                : Integer;
           tau,tausq           : single;
           ts1,ts2,tn0         : word;
           ts0,tsf,tn          : integer;
           ParType             : Word;  {0: percentages; 1: absolute times}
           ParStart            : Word;
         BEGIN
           ParType:=0;
           IF (Filt = F_SlopeFilt2) THEN ParType:=1;
           IF (Filt = F_SlopeFilt) THEN ParType:=0;

           ParStart:=1;
           SetFiltParams(ParType,ParStart,LowFreq,HighFreq);

           ParStart:=3;
           SetTaperParams(ParType,ParStart,ts1,ts2,tn0);

           WITH Hmap DO
             BEGIN
               N := TRUNC (Ln(Tqty-1) / Ln(2)) + 1;
               N := N + Sets.FreqWind;
               N := ROUND ( EXP(N*Ln(2)) );
               IF (N > F_MaxEl) THEN N := F_MaxEl;
               st  := (N - Tqty) DIV 2;
               fin := st + Tqty - 1;

               {Zero Padding}
               FOR i := 0 TO N-1 DO
                 Four[i] := Compl0;
               FOR i := 1 TO (Tqty) DO
                 Four[(st-1)+i].Rl := Hst^[i];

               {Exponential Taper - start side}
               tn:=tn0;
               IF (tn>ts1) THEN tn:=ts1;
               ts0:=ts1-tn;
               FOR i := 0 TO (ts0-1) DO
                   Four[st+i].Rl := 0;
               FOR i := ts0 TO ts1 DO
                 BEGIN
                   tau   := ((i-ts1)/(tn+1)); tausq := tau*tau;
                   Four[st+i].Rl := Four[st+i].Rl * exp(-10*tausq);
                 END;

               {Exponential Taper - finish side}
               tn:=tn0;
               IF (tn>(fin-st-ts2)) THEN tn:=(fin-st-ts2);
               tsf:=ts2+tn;
               FOR i := ts2 TO tsf DO
                 BEGIN
                   tau   := ((i-ts2)/(tn+1)); tausq := tau*tau;
                   Four[st+i].Rl := Four[st+i].Rl * exp(-10*tausq);
                 END;
               FOR i := (tsf+1) TO (fin-st) DO
                   Four[st+i].Rl := 0;

               FFT (Four, N, F_Trform);
               IF (Filt <> F_NoFilt)
                  THEN F_Filter (Four, Filt, N, TRUNC (N*LowFreq), TRUNC (N*HighFreq) );

               IF (Inverse)
                  THEN BEGIN
                         FFT (Four, N, F_Inverse);
                         Tqty := N; Tstart:=Tstart-(st*dT); Time:=Tstart+((Tqty-1)*dT);
                         FOR i := 1 TO Tqty DO
                           Hst^[i] := Four[i-1].Rl;
                       END
                  ELSE BEGIN
                         CASE Repr OF
                           F_ampl  :  BEGIN
                                        FOR i := 1 TO N DO
                                          BEGIN
                                            IF ( Arctan0 (Four[i-1].Im, Four[i-1].Rl) >= 0)
                                               THEN Sign := 1
                                               ELSE Sign := -1;
                                            Hst^[i] := {Sign *} SQRT ( SQR(Four[i-1].Rl) + SQR(Four[i-1].Im) );
                                            Hst^[i] := Hst^[i]*dT; {The FT should be divided by maximum frequency}
                                          END;
                                        GRs.ScalT := LogScal;
                                      END;
                           F_phase : FOR i := 1 TO N DO
                                       BEGIN
                                         Hst^[i] := Arctan0 (Four[i-1].Im, Four[i-1].Rl) * 180/PI;
                                         {Enforce result in range 0 to +360}
                                         IF (Hst^[i] < 0) THEN Hst^[i] := Hst^[i]+360;
                                         {Then force positively increasing phase}
                                         {03/07/07: Remove if you want unadulterated phase angle}
                                         IF (i>1) THEN
                                            IF (Hst^[i]<Hst^[i-1]) THEN
                                               Hst^[i] := INT((Hst^[i-1]-Hst^[i])/360+1)*360 + Hst^[i];
                                         IF (i>1) THEN
                                            IF ((Hst^[i]-Hst^[i-1]) > 360) THEN Hst^[i]:=Hst^[i]-360;

(*                                         Hst^[i] := Arctan0 (Four[i-1].Im, Four[i-1].Rl) * 180/PI;
                                         {Enforce result in range 0 to -360}
                                         IF (Hst^[i] >= 0) THEN Hst^[i] := Hst^[i]-360;
                                         {Then force negatively (decreasing) phase}
                                         {03/07/07: Remove if you want unadulterated phase angle}
                                         IF (i>1) THEN
                                            IF (Hst^[i]>Hst^[i-1]) THEN
                                               Hst^[i] := INT((Hst^[i-1]-Hst^[i])/360-1)*360 + Hst^[i];
                                         IF (i>1) THEN
                                            IF ((Hst^[i]-Hst^[i-1]) <= -360) THEN Hst^[i]:=Hst^[i]+360; *)
                                       END;
                         END; {Case}
                         dT := 1/dT/N;  Tqty := (N DIV 2)+1;
{                        Tqty := N;     Time := (Tqty-1)*dT;}
                         Tstart:=0;     Time := (Tqty-1)*dT;
                         GRs.xstr := 'Hz';
                    END; {If direct transform}
             END; {With Hmap}
         END; {Fourier}


       PROCEDURE Fourier2 (Hst1: HistPtr;
                           VAR Four: FourArray;
                           VAR N: Integer);
         VAR
           i,st,fin  : Integer;
           LowFreq, HighFreq   : Single;
           LowTaper, HighTaper,
             MaxTaper          : Single;
           Ampl, ang           : single;
           pi2                 : single;
           tau,tausq           : single;
           tn,ts1,ts0          : integer;
         BEGIN
           LowFreq := Sets.LowFreq / 100;
           HighFreq := Sets.HighFreq / 100;
           LowTaper := Sets.LowTaper / 100;
           HighTaper := Sets.HighTaper / 100;
           MaxTaper := Sets.MaxTaper / 100;
           pi2 := 2*pi;
           WITH Hmap DO
             BEGIN
               N := TRUNC (Ln(Tqty-1) / Ln(2)) + 1;
               N := N + Sets.FreqWind;
               N := ROUND ( EXP(N*Ln(2)) );
               IF (N > F_MaxEl)
                  THEN N := F_MaxEl;
               st  := (N - Tqty) DIV 2;
               fin := st + Tqty - 1;

(* Old Taper  {Zero Padding}
               FOR i := 0 TO N-1 DO
                 Four[i] := Compl0;
               FOR i := 1 TO (Tqty) DO
                 Four[(st-1)+i].Rl := Hst1^[i];

               {Exponential Taper - start side}
               tn:= TRUNC (Tqty*LowTaper);
               ts1:=st+tn;
               FOR i := st TO ts1 DO
                 BEGIN
                   tau   := ((i-ts1)/(tn+1)); tausq := tau*tau;
                   Four[i].Rl := Four[i].Rl * exp(-10*tausq);
                 END;

               {Exponential Taper - finish side}
               tn:= TRUNC (Tqty*(1.0-HighTaper));
               ts1:=fin-tn;
               FOR i := ts1 TO fin DO
                 BEGIN
                   tau   := ((ts1-i)/(tn+1)); tausq := tau*tau;
                   Four[i].Rl := Four[i].Rl * exp(-10*tausq);
                 END;
*)

               {Zero Padding}
               FOR i := 0 TO N-1 DO
                 Four[i] := Compl0;
               FOR i := 1 TO (Tqty) DO
                 Four[(st-1)+i].Rl := Hst1^[i];

               {Exponential Taper - start side}
               ts1:= TRUNC (Tqty*LowTaper);
               tn := TRUNC (Tqty*MaxTaper);
               IF (tn>ts1) THEN tn:=ts1;
               ts0:=ts1-tn;
               FOR i := 0 TO (ts0-1) DO
                   Four[st+i].Rl := 0;
               FOR i := ts0 TO ts1 DO
                 BEGIN
                   tau   := ((i-ts1)/(tn+1)); tausq := tau*tau;
                   Four[st+i].Rl := Four[st+i].Rl * exp(-10*tausq);
                 END;

               {Exponential Taper - finish side}
               ts1:= st+TRUNC (Tqty*HighTaper);
               tn := TRUNC (Tqty*MaxTaper);
               IF (tn>(fin-ts1)) THEN tn:=(fin-ts1);
               ts0:=ts1+tn;
               FOR i := ts1 TO ts0 DO
                 BEGIN
                   tau   := ((i-ts1)/(tn+1)); tausq := tau*tau;
                   Four[i].Rl := Four[i].Rl * exp(-10*tausq);
                 END;
               FOR i := (ts0+1) TO fin DO
                   Four[i].Rl := 0;


               FFT (Four, N, F_Trform);
               FOR i := 1 TO N DO
                 BEGIN
                   Ampl := SQRT ( SQR(Four[i-1].Rl) + SQR(Four[i-1].Im) );
                   ang  := Arctan0 (Four[i-1].Im, Four[i-1].Rl);

                   {assumption - (a) phase response monotonic}
{                   IF (ang < 0) THEN ang := ang + pi2;
                   IF (i>0) THEN
                      IF (ang<Four[i-2].Im) THEN
                         ang := INT((Four[i-2].Im-ang)/pi2+1)*pi2 + ang;}

                   Four[i-1].Rl := Ampl;
                   Four[i-1].Im := Ang;
                 END;

             END; {With Hmap}
         END; {Fourier2}


       PROCEDURE InvFourier (Hst1,Hst2: HistPtr);
           {Takes amplitude (Hst^) and phase (Hst2^) info, and inverts to time domain}
         VAR
           N,i,st,fin        : Integer;
           ang               : single;
           Four              : FourArray;
         BEGIN
           WITH Hmap DO
             BEGIN
               N:=(Tqty-1)*2;                  {Data-set length was modified by 'F' and 'P' functions}
               FOR i := 0 TO (N-1) DO
                 BEGIN
                   ang := Hst2^[i+1]*PI/180;
                   Four[i].Rl := Hst1^[i+1]*cos(ang);
                   Four[i].Im := Hst1^[i+1]*sin(ang);
                 END;
{               FOR i := 0 TO ((N DIV 2)-1) DO
                 BEGIN
                   ang := Hst2^[i]*PI/180;
                   Four[i].Rl := Hst1^[i]*cos(ang);
                   Four[i].Im := Hst1^[i]*sin(ang);
                   Four[N-1-i].Rl := Four[i].Rl;
                   Four[N-1-i].Im := Four[i].Im;
                 END;}
               FFT (Four, N, F_Inverse);

               dT := 1/dT/N;  Tqty := N;
               Tstart:=0;     Time := (Tqty-1)*dT;
               GRs.xstr := 's';
               GRs.ScalT := LinScal;
               FOR i := 1 TO Tqty DO
                 Hst1^[i] := Four[i-1].Rl;
             END; {With Hmap}
         END; {InvFourier}

       PROCEDURE MapWtoG (Hst1,Hst2: HistPtr;
                          VAR Four1 : FourArray;
                          Tqty:integer;
                          VAR dT,Time:single);
         VAR
           C0,dg,wi,gj,gi,gip : single;
           i,j : integer;
         BEGIN
          {Average a few early values of C(w)}
           C0 := 0;
           FOR i:=3 TO 8 DO
             C0 := C0+Four1[i-1].Rl;
           C0:=C0/6;
           IF (params[2]<>0) THEN C0:=params[2];
           dg:=dT/C0;
           i:=1;
           wi:=0;   {time freq}
           gj:=0;   {spacial freq}
           Hst2^[1]:=Hst1^[1];
           gi:=0;
           gip:=gi;
           FOR j := 2 TO Tqty DO
             BEGIN
               gj:=gj+dg;
               WHILE ((gi<gj) AND (i<Tqty)) DO
                 BEGIN
                   inc(i); wi:=wi+dT; gip:=gi; gi:=wi/Four1[i-1].Rl;
                 END;

               IF (gip=gi)
                  THEN Hst2^[j]:= Hst2^[i-1]
                  ELSE Hst2^[j]:=   (gj-gip)/(gi-gip)*Hst1^[i]
                                  + (gi-gj)/(gi-gip)*Hst1^[i-1];
             END; {For}
           FOR j := 1 TO Tqty DO
             Hst1^[j]:=Hst2^[j];
           dT:=dG;
           Time := (Tqty-1)*dT;
         END; {MapWtoG}



       PROCEDURE PhaseVel (Hst1,Hst2: HistPtr; tch:char);
           {Takes amplitude ratio (Hst^) and phase difference (Hst2^), and calcs phase velocity}
           {Params: param1=Tdiff:  Time-diff for spatial waveforms -
                                   (or space-diff for time waveforms)
                    param2=wave-speed: used to convert from freq to wavenumber
                    param3=attenl: Unit length for attenuation}
         VAR
           N,i,st,fin        : Integer;
           ang               : single;
{           Four1, Four2      : FourArray;}
           ww, pi2           : single;
           pdiff,rr          : single;
           tdiff,attenl      : single;
         BEGIN
(*           pi2 := 2*pi;
           tdiff := params[1];
           attenl := abs(params[3]);
           IF (attenl<=1e-12) THEN attenl:=1;
           Fourier2 (Hst1,Four1,N);
           Fourier2 (Hst2,Four2,N);
           FOR i := 1 TO N DO
             BEGIN
               IF (Four1[i-1].Rl <> 0)
                  THEN Hst1^[i] := Four2[i-1].Rl / Four1[i-1].Rl
{                  ELSE Hst1^[i] := Four2[i-1].Rl;}
                  ELSE IF (i>1)
                          THEN Hst1^[i] := Hst1^[i-1]
                          ELSE Hst1^[i] := 1;

               {note, needs to be made -ve - phase shouldn't be increasing - check arctan function in fourier2?}
               Hst2^[i] := - (Four2[i-1].Im - Four1[i-1].Im);
               IF (Hst2^[i] < 0) THEN Hst2^[i]:=Hst2^[i]+pi2;

               {assumption - (b) change in 'phase difference' from one freq to next less than +-pi}
{               IF (i>1) THEN BEGIN
                   IF (Hst2^[i-1]<0)
                      THEN Hst2^[i] := INT(Hst2^[i-1]/pi2-1)*pi2+Hst2^[i]
                      ELSE Hst2^[i] := INT(Hst2^[i-1]/pi2)  *pi2+Hst2^[i];
                   IF ((Hst2^[i]-Hst2^[i-1]) >  pi) THEN Hst2^[i]:=Hst2^[i]-pi2;
                   IF ((Hst2^[i]-Hst2^[i-1]) < -pi) THEN Hst2^[i]:=Hst2^[i]+pi2;
               END;} {If}

               {assumption - (c) phase difference is monotonic
                         and (d) increment from one freq to next less than pi}
               {assumption - (e) change in 'phase difference' from one freq to next less than zero}
               IF (i>1) THEN BEGIN
                   pdiff :=  pi;
{                   pdiff := 0;}
                   WHILE ( (Hst2^[i]-Hst2^[i-1]) >= pdiff) DO
                     Hst2^[i]:=Hst2^[i]-pi2;
                   pdiff := (Hst2^[i]-Hst2^[i-1]);
               END; {If}

             END;

           WITH Hmap DO
             BEGIN
{               dT := 1/dT/N;  Tqty := (N DIV 2)+1;
               Tstart:=0;     Time := (Tqty-1)*dT;}
               {Use full sample frequency:   fs = 2*fn the Nyquist freq}
               {Normally just show freq up to the Nyquist freq as even function}
               dT := 1/dT/N;    Tqty := N;
               Tstart:=0;       Time := (Tqty-1)*dT;
               N := Tqty;
               GRs.xstr := '#';
               GRs.ScalT := LinScal;

               FOR i := 0 TO (N-1) DO
                 BEGIN
                   IF (i=0)
                      THEN ww:=N*dT*pi2 ELSE ww := i*dT*pi2;
                   Four1[i].Rl := Hst2^[i+1];
{                   Four1[i].Im := ln(Hst1^[i+1]);}
                   IF (Hst1^[i+1]<=1E-20)
                      THEN Four1[i].Im := -50
                      ELSE Four1[i].Im := ln(Hst1^[i+1]);
{                   IF (tch IN ['W','w','Z','z']) THEN Four1[i].Im := 0;}
                   WITH Four1[i] DO
                     BEGIN
                       rr := (Rl*Rl + Im*Im);
                       CASE tch OF
                         'V','v' : BEGIN
                                     Rl := -Rl/ww/tdiff;
                                     Im := Im/ww/tdiff;
                                   END;
                         'W','w','Z','z' : BEGIN
                                     IF (rr=0) THEN Rl:=0 ELSE Rl:=(-Rl/rr)*ww*tdiff;
                                     IF (rr=0) THEN Im:=0 ELSE Im:=(-Im/rr)*ww*tdiff;
                                   END;
                         'A','a' : BEGIN
                                     IF (rr=0) THEN Rl:=0 ELSE Rl:=(-Rl/rr)*ww*tdiff;
{                                    NO! - Im :=1 + (exp(Im)-1)/tdiff*attenl;}  {% Attenuation per 'attenl' metres}
                                     Im :=exp((attenl/tdiff)*Im);  {% Attenuation per 'attenl' metres}
                                   END;
                         'B','b' : BEGIN
                                     Rl := Rl * 180/pi;
                                   END;
                       END; {Case}
                     END; {with}

                 END; {For}

               FOR i := 1 TO Tqty DO
                 WITH Four1[i-1] DO
                    CASE tch OF
                     {Phase velocities}
                      {Note - Imaginary part <<< Real (1e6), can ignore}
{                      'v' : Hst1^[i] := SQRT(Rl*Rl + Im*Im);}
                     'v' : Hst1^[i] := Rl;
{                     'w' : Hst1^[i] := SQRT(Rl*Rl + Im*Im);}
                      'w','z' : Hst1^[i] := Rl;
                      'A','a' : Hst1^[i] := Im;
                      'B','b' : Hst1^[i] := Rl;

                     {Group velocities}
{                      'V' : Hst1^[i] := Rl + (i*dT)*(Four1[i].Rl-Rl) / Dt; simplify}
                      'V' : IF (i=Tqty)
                               THEN Hst1^[i] := Hst1^[i-1]
                               ELSE Hst1^[i] := Rl + (i)*(Four1[i].Rl-Rl);
{                      'W' : Hst1^[i] := Rl*RL / (Rl - (i*dT)*(Four1[i].Rl-Rl) / Dt); simplify}
                      'W','Z' : IF (i=Tqty)
                               THEN Hst1^[i] := Hst1^[i-1]
                               ELSE Hst1^[i] := Rl*Rl / (Rl - (i)*(Four1[i].Rl-Rl));
                    END; {Case}

               {Transform c(w) to c(k), assume ci*wi=ki is monotonic!}
               IF (tch IN ['z','Z','A','B'])
                  THEN MapWtoG(Hst1,Hst2,Four1,Tqty,dT,Time);

               {Try map atten, so atten per 10 cycles, rather than 10 metres}
               {Needs more thought!}
{                IF (tch IN ['A','a']) THEN
                  FOR i := 1 TO Tqty DO
                    Hst1^[i] := 1+(Hst1^[i]-1)*i*dT*10; }

               {If phase vel, convert metres to Hz}
{               CASE tch OF
                 'V','v' : BEGIN
                             dT := dT*Hst1^[3]; Tstart:=0; Time:=(Tqty-1)*dT;
                           END;
               END;} {Case}

             END; {With Hmap}
*)         END; {PhaseVel}

       PROCEDURE Func1 (f: Char);
           {Function of one variable:
              data-set given after the function in brackets.
              if no brackets, then the accumulated data-set Hst^ is used}
         VAR
           Prev1, Prev2        : Single;
           Mult, power, tShift : Single;
           Limit               : Single;
           t, Shift            : Integer;

           Hst2                : HistPtr;
         BEGIN
           Delete (Pstring, 1, 1);
           ParseParam;
           IF (Pstring [1] = '(')
              THEN ParseBrackets(Hst);
           CASE f OF
             {Invert - i.e. exchange horizontal from x2 to x1}
             'X', 'x' : BEGIN
                          NEW (Hst2);
                          FOR t := 1 TO Hmap.Tqty  DO
                            Hst2^[t] := Hst^[Hmap.Tqty+1-t];
                          FOR t := 1 TO Hmap.Tqty  DO
                            Hst^[t] := Hst2^[t];
                          DISPOSE (Hst2);
                        END;
             'M', 'm' : BEGIN {Multiply by constant}
                          IF (ParamCount >= 1)
                             THEN Mult := Params[1]
                             ELSE Mult := 1;
                          FOR t := 1 TO Hmap.Tqty  DO
                            Hst^[t] := Hst^[t] * Mult;
                        END;
             '#'      : BEGIN {Add a constant}
                          IF (ParamCount >= 1)
                             THEN Mult := Params[1]
                             ELSE Mult := 0;
                          FOR t := 1 TO Hmap.Tqty  DO
                            Hst^[t] := Hst^[t] + Mult;
                        END;
             'I', 'i' : BEGIN
                          Prev2 := Hst^[1];
                          Hst^[1] := 0;
                          FOR t := 2 TO Hmap.Tqty  DO
                            BEGIN
                              Prev1 := Hst^[t];
                              Hst^[t] := Hst^[t-1] + Prev2 * Hmap.Dt;
                              Prev2 := Prev1;
                            END;

{                             Hst^[1] := Hst^[1] * Hmap.Dt;
                             FOR t := 2 TO Hmap.Tqty  DO
                               Hst^[t] := Hst^[t-1] + Hst^[t] * Hmap.Dt;}

                        END;
             'D', 'd' : BEGIN
                          FOR t := 1 TO (Hmap.Tqty-1)  DO
                            Hst^[t] := (Hst^[t+1] - Hst^[t]) / Hmap.Dt;
                          Hst^[Hmap.Tqty] := Hst^[Hmap.Tqty-1];

{                             Prev2 := Hst^[1];
                             Hst^[1] := Hst^[1] / Hmap.Dt;
                             FOR t := 2 TO Hmap.Tqty  DO
                               BEGIN
                                 Prev1 := Hst^[t];
                                 Hst^[t] := (Hst^[t] - Prev2) / Hmap.Dt;
                                 Prev2 := Prev1;
                               END;}

                        END;
             's'      : BEGIN {Just shift start and end time}
                          IF (ParamCount >= 1)
                             THEN tShift := Params[1]
                             ELSE tShift := 0;
                          Hmap.Tstart := Hmap.Tstart+tShift;
                          Hmap.Time   := Hmap.Time+tShift;
                        END;
             'S'      : BEGIN  {Actually shift values}
                          IF (ParamCount >= 1)
                             THEN Shift := TRUNC (Params[1])
                             ELSE Shift := 0;
                          IF ( ABS(Shift) > Hmap.Tqty)
                             THEN Shift := 0;
                          IF (Shift >= 0)
                             THEN BEGIN
                                    FOR t := Hmap.Tqty DOWNTO (Shift+1) DO
                                      Hst^[t] := Hst^[t-Shift];
                                    FOR t := Shift DOWNTO 1 DO
                                      Hst^[t] := Hst^[1];
                                  END
                             ELSE BEGIN
                                    Shift := ABS(Shift);
                                    FOR t := 1 TO (Hmap.Tqty-Shift) DO
                                      Hst^[t] := Hst^[t+Shift];
                                    FOR t := (Hmap.Tqty-Shift+1) TO Hmap.Tqty DO
                                      Hst^[t] := Hst^[Hmap.Tqty];
                                  END;
                        END;
             'E', 'e' : BEGIN {raise to the power of - use exponent}
                          IF (ParamCount >= 1)
                             THEN power := Params[1]
                             ELSE power := 0.5;
                          FOR t := 1 TO Hmap.Tqty  DO
                            Hst^[t] := POW (Hst^[t], power);
                        END;
             'J', 'j' : Extract; {Extract part of waveform - apply taper}
             'L', 'l' : BEGIN {Limit phase angles etc to increments of +-param/2}
                          Limit := Params[1]-Params[2];
                          Prev1 := Hst^[1];
                          FOR t := 2 TO Hmap.Tqty  DO
                            BEGIN
                              WHILE (Hst^[t]-Prev1 > Params[1]) DO
                                Hst^[t]:=Hst^[t]-Limit;
                              WHILE (Hst^[t]-Prev1 < Params[2]) DO
                                Hst^[t]:=Hst^[t]+Limit;
                              Prev1 := Hst^[t];
                            END;
                        END;

             'P'      : Fourier (F_Trform,  F_NoFilt,   F_phase);  {Fourier T'form - phase spectrum}
             'p'      : Fourier (F_Trform,  F_StepFilt, F_phase);  {Fourier T'form - phase spectrum, filtered}
             'F'      : Fourier (F_Trform,  F_NoFilt,   F_ampl);   {Fourier T'form - amplitude spectrum}
             'f'      : Fourier (F_Trform,  F_StepFilt, F_ampl);   {Fourier T'form - amplitude spectrum, filtered}
             'T'      : Fourier (F_Inverse, F_NoFilt,   F_ampl);   {Inverse F. T'form, of Fourier T'form, as test.}
             't'      : Fourier (F_Inverse, F_StepFilt, F_ampl);   {Inverse F. T'form, of Filtered Fourier Transform, as test}
           END; {Case}
         END;

       PROCEDURE Func2 (f: Char);
           {Function of two variables:
              assumes 1st data-set is accumulated in Hst^
              2nd data-set given after the function}
         VAR
           t                : Word;
           Hst2             : HistPtr;
           CosThet, SinThet : Single;
           Sign             : Integer;
          {temp variables for rotation}
           nn               : Integer;
           diff             : Single;
         BEGIN
           Delete (Pstring, 1, 1);
           ParseParam;
           NEW (Hst2);
           Parse (Hst2, Pstring);
           CASE f OF
             '+' : FOR t := 1 TO Hmap.Tqty  DO
                     Hst^[t] := Hst^[t] + Hst2^[t];
             '-' : FOR t := 1 TO Hmap.Tqty  DO
                     Hst^[t] := Hst^[t] - Hst2^[t];
             '*' : FOR t := 1 TO Hmap.Tqty  DO
                     Hst^[t] := Hst^[t] * Hst2^[t];
             '/' : FOR t := 1 TO Hmap.Tqty  DO
                     IF (Hst2^[t] <> 0)
                        THEN Hst^[t] := Hst^[t] / Hst2^[t];
             'c',
             'C' : BEGIN
                     CosThet := COS (Params[1] * PI / 180);
                     SinThet := SIN (Params[1] * PI / 180);
                     FOR t := 1 TO Hmap.Tqty  DO
                         Hst^[t] := Hst^[t]*CosThet + Hst2^[t]*SinThet;
                   END;
             'R' : FOR t := 1 TO Hmap.Tqty  DO
                     BEGIN
{                       IF ( ArcTan0 (Hst2^[t], Hst^[t]) >= 0)
                          THEN Sign := 1
                          ELSE Sign := -1;}
                       Hst^[t] := {Sign *} SQRT ( SQR(Hst^[t]) + SQR(Hst2^[t]) );
                     END;
             'r' : FOR t := 1 TO Hmap.Tqty  DO
                     BEGIN
                       Hst^[t] := ArcTan0 (Hst2^[t], Hst^[t]) * 180 / PI;
{                       IF (Hst^[t] < 0) THEN Hst^[t] := Hst^[t] + 2*PI;
                       IF (t > 1)
                         THEN BEGIN
                          nn   := TRUNC(Hst^[t-1]) DIV 360;  diff := TRUNC(Hst^[t-1]) MOD 360;
                          IF (Hst^[t-1] < 0)  THEN BEGIN  nn:=nn-1; diff:=diff+360; END;
                          diff := diff-Hst^[t];
                          IF (diff > 180) THEN nn:=nn+1;
                          IF (diff < -180) THEN nn:=nn-1;
                          Hst^[t] := Hst^[t] + nn*360;
                         END;}

{                       IF (Hst^[t] < 0)
                          THEN Hst^[t] := Hst^[t] + 180;
                       IF (  (t > 1) AND ( (Hst^[t] - Hst^[t-1]) > 90)   )
                          THEN Hst^[t] := Hst^[t] - 180;
                       IF (  (t > 1) AND ( (Hst^[t] - Hst^[t-1]) < -90)   )
                          THEN Hst^[t] := Hst^[t] + 180;}
                     END;
           END; {Case}
           DISPOSE (Hst2);
           Pstring := '';
         END;

       PROCEDURE Func2b (f: Char);
           {Function of two variables:
              ignores data-set accumulated in Hst^
              1st and 2nd data-set given after the function (e.g. "G([1])([2])" )}
         VAR
           t                : Word;
           Hst2             : HistPtr;
           CosThet, SinThet : Single;
           Sign             : Integer;
         BEGIN
           Delete (Pstring, 1, 1);
           ParseParam;
           ParseBrackets(Hst);
           NEW (Hst2);
           ParseBrackets(Hst2);
{           Parse (Hst2, Pstring);}
           CASE f OF
             'G','g' : InvFourier (Hst,Hst2);   {Inverts, given amplitude and phase}
             'V','v' : PhaseVel   (Hst,Hst2,f);   {phase velocity from spatial waveforms}
             'W','w' : PhaseVel   (Hst,Hst2,f);   {phase velocity from time waveforms}
             'Z','z' : PhaseVel   (Hst,Hst2,f);   {phase velocity from time waveforms - converts to spatial frequency}
             'A','a' : PhaseVel   (Hst,Hst2,f);   {Temp}
             'B','b' : PhaseVel   (Hst,Hst2,f);   {Temp}
(*             'H','h' : BEGIN                  {Calculates transfer function (amplit)}
                         Fourier (F_Trform,  F_NoFilt,   F_ampl);
                         Fourier (F_Trform,  F_NoFilt,   F_ampl);
                       END;*)
           END; {Case}
           DISPOSE (Hst2);
           Pstring := '';
         END;


       PROCEDURE Func2c (f: Char);
           {Function of two variables (extended "$" commands):
              assumes 1st data-set is accumulated in Hst^
              2nd data-set given after the function}
         VAR
           t,t1,PP,SS       : Word;
           h,s              : Single;
           Hst2, Hst3             : HistPtr;
         BEGIN
           Delete (Pstring, 1, 1);
           ParseParam;
           NEW (Hst2); NEW (Hst3);
           Parse (Hst2, Pstring);
           CASE f OF
             'g' : {Calculates sub-function "g" of boundary reflection}
                   {New function g, overwrites x2, and later x1, the supplied differences}
                BEGIN
                  PP := ROUND(Params[1]); SS := ROUND(Params[2]); s:=0;
                  FOR t := 1 TO Hmap.Tqty  DO
                    BEGIN
                      h :=0;
                      t1 := t+PP;    IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h+Hst2^[t1]; {+x2(P)}
                      t1 := t+PP-SS; IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h-Hst^[t1];  {-x1(P-S)}
                      t1 := t+PP-SS; IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h+Hst3^[t1]; {+g(P-S)}
                      t1 := t-PP;    IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h+Hst3^[t1]; {+g(-P)}
                      t1 := t-SS;    IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h-Hst3^[t1]; {-g(-S)}
                      h:=h-0.5*s; {damp?}
                      Hst3^[t] := h; {Needs to first be stored in Hst2}
                      s := s+h;
                    END;
                  FOR t := 1 TO Hmap.Tqty  DO
                    Hst^[t]:=Hst3^[t]
                END;
             'h' : {Calculates sub-function "h" of boundary reflection}
                   {New function h, overwrites x2, and later x1, the supplied differences}
                BEGIN
                  PP := ROUND(Params[1]); SS := ROUND(Params[2]);
                  FOR t := 1 TO Hmap.Tqty  DO
                    BEGIN
                      h :=0;
                      t1 := t+PP;    IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h-Hst2^[t1]; {-x2(P)}
                      t1 := t;       IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h+Hst^[t1];  {+x1()}
                      t1 := t+PP-SS; IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h+Hst3^[t1]; {+h(P-S)}
                      t1 := t-SS;    IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h+Hst3^[t1]; {+h(-S)}
                      t1 := t+PP-2*SS; IF ((t1>=1) AND (t1<=Hmap.Tqty)) THEN h:=h-Hst3^[t1]; {-h(P-2S)}
                      Hst3^[t] := h; {Needs to first be stored in Hst2}
                    END;
                  FOR t := 1 TO Hmap.Tqty  DO
                    Hst^[t]:=Hst3^[t]
                END;

             '*' : {Convolution - done in time domain}
                BEGIN
                  FOR t := 1 TO Hmap.Tqty  DO
                    BEGIN
                      h :=0;
                      FOR t1 := 1 TO Hmap.Tqty  DO
                        IF (t>t1) THEN h:=h+(Hst^[t1]*Hst2^[t-t1])*Hmap.Dt;
                      Hst3^[t] := h;
                    END;
                  FOR t := 1 TO Hmap.Tqty  DO
                    Hst^[t]:=Hst3^[t]
                END; { '*' convolution }

             'G' : {Calculates sub-function "G" of boundary reflection}
                   {New function g, overwrites x2, and later x1, the supplied differences}
                BEGIN
                  PP := ROUND(Params[1]); SS := ROUND(Params[2]);
                  FOR t := 1 TO Hmap.Tqty  DO
                    BEGIN
                      h :=0;
{                      IF (t>PP) THEN h:=sqrt(ABS(Hst^[t]*Hst2^[t-PP]));
                      IF (Hst^[t]<0) THEN h:=-h;}
                      IF (t<=(Hmap.Tqty-PP)) THEN h:=0.5*(Hst^[t]+Hst2^[t+PP]);
                      Hst3^[t] := h;
                    END;
                  FOR t := 1 TO Hmap.Tqty  DO
                    IF (t<=PP) THEN Hst^[t]:=Hst3^[t]
                               ELSE Hst^[t]:=Hst3^[t]+Hst^[t-PP];
{                    Hst^[t]:=Hst3^[t];}
                END; {'G'}

           END; {Case}
           DISPOSE (Hst3); DISPOSE (Hst2);
           Pstring := '';
         END; {Func2c}


       PROCEDURE Func1b (f: Char);
           {Function of one variable: - used for "&" extended functions
              data-set given after the function in brackets.
              if no brackets, then the accumulated data-set Hst^ is used}
         VAR
           t                   : Integer;
           alpha, Dsmall       : Single;
           Tqty2               : Integer;
           Hst2                : HistPtr;
           Xi,Di,Ti            : Single;
           DecVal              : Integer;
           flow, fhi           : Single;
           ford, f0phase       : Integer;
           typ,rtyp            : Integer;
           asecs,Ds,st,damp    : single;
           Nns, st_i           : Integer;
           afact               : single;  {derive of damping}
           Qval,Cval,Lval      : single;
           rcomp,rhnum,ri,rj,rk   : integer;
           r_gam,r_cgam,r_sgam,r_thet,r_cthet,r_sthet : single;

         BEGIN
           Delete (Pstring, 1, 1);
           ParseParam;
           IF (Pstring [1] = '(')
              THEN ParseBrackets(Hst);
           Tqty2 := Hmap.Tqty;
           CASE f OF
             {x(t) used in Tp calculation}
             'x', 'X' : BEGIN
                     {centre "start" position for function}
                     IF (ParamCount >= 1) THEN st:=Params[1] ELSE st:=10;
                     IF (st <= 0) THEN st:=10;
                     st_i := TRUNC(st/Hmap.Dt)+1;
                     IF (st_i<20) THEN BEGIN st_i:=20; st:=(st_i-1)*Hmap.Dt; END;
                     IF (st_i>= (MaxHistSz div 2)) THEN BEGIN st_i:=((MaxHistSz div 2)-1); st:=(st_i-1)*Hmap.Dt; END;
                     {window length}
                     IF (ParamCount >= 2) THEN asecs:=Params[2] ELSE asecs:=10;
                     IF (asecs <= 0) THEN asecs:=10;
                     alpha := exp (ln(0.1) / (asecs/Hmap.Dt));
                     {type 0 - truncated; 1 - even}
                     IF (ParamCount >= 3) THEN typ:=trunc(Params[3]) ELSE typ:=0;
                     IF (typ > 1) THEN typ:=1;

                     damp:=1.0;
                     FOR t := st_i DOWNTO 1 DO
                         BEGIN  Hst^[t] := Hst^[t]*Damp; damp:=damp*alpha;  END;
                     IF (typ=0) THEN BEGIN Hmap.Tqty:=st_i; Hmap.Time:=st;  END;
                     IF (typ=1)
                        THEN BEGIN
                               FOR t := (st_i-1) DOWNTO 1 DO  Hst^[st_i+st_i-t]:=Hst^[t];
                               Hmap.Tqty:=2*st_i-1; Hmap.Time:=(Hmap.Tqty-1)*Hmap.dt;
                             END;
                   END; {'x/X': x(t) for Tp calc}

             'D','d':  {Decimate - param n - desample by n times}
                   BEGIN
                     IF (ParamCount >= 1) THEN DecVal:=TRUNC(Params[1]) ELSE DecVal:=1;
                     IF (DecVal < 1) THEN DecVal:=1;
                     IF (DecVal > 1) THEN
                        BEGIN
                          Tqty2:=Hmap.Tqty DIV DecVal;
                          FOR t := 1 TO Tqty2  DO
                            Hst^[t] := Hst^[t*DecVal];
                          Hmap.Tqty := Tqty2;
                          Hmap.dt   := Hmap.dt*DecVal;
                          Hmap.Tstart := Hmap.Tstart;
                          Hmap.Time := Hmap.Tstart + Hmap.Tqty*Hmap.dt;
                        END;
                   END;


             'W', 'w' : Extract; {Window - Extract part of waveform, apply taper}
             {Test fd filts - p,q,Q letters}
             'F'  : Fourier (F_Inverse, F_SlopeFilt2, F_ampl);   {Filtered waveform - Inverse FT, of Filtered FT}
             'P'  : Fourier (F_Trform,  F_SlopeFilt2, F_ampl);   {Fourier T'form - amplitude spectrum, filtered}
             'p'  : Fourier (F_Trform,  F_SlopeFilt2, F_phase);  {Fourier T'form - phase spectrum, filtered}

             'f' :  {Butterworth filter option - zero phase}
                   BEGIN
                     IF (ParamCount >= 1) THEN flow:=Params[1] ELSE flow:=0;
                     IF (ParamCount >= 2) THEN fhi:=Params[2]  ELSE fhi:=0;
                     IF (ParamCount >= 3) THEN ford:=trunc(Params[3])  ELSE ford:=4;
                     IF (ParamCount >= 4) THEN f0phase:=trunc(Params[4])  ELSE f0phase:=1; {default 1 is zero phase}
                     bp_bworth (Hst, Hmap.Dt, Hmap.Tqty, flow, fhi, ford, f0phase);
                   END; {'F' butterworth filter option}

             'q'  :  {Apply Q to waveform}
                   BEGIN
                     IF (ParamCount >= 1) THEN Qval:=Params[1] ELSE Qval:=1e6;
                     IF (ParamCount >= 2) THEN Cval:=Params[2] ELSE Cval:=6000;
                     IF (ParamCount >= 3) THEN Lval:=Params[3] ELSE Lval:=1;
                     ApplyQ (Qval,Cval,Lval);
                   END; {'q' apply Q to a waveform}

             'Q'  :  {Apply Q to 3 waveforms and calc Vabs}
                   BEGIN
                     IF (ParamCount >= 1) THEN Qval:=Params[1] ELSE Qval:=1e6;
                     IF (ParamCount >= 2) THEN Cval:=Params[2] ELSE Cval:=6000;
                     IF (ParamCount >= 3) THEN Lval:=Params[3] ELSE Lval:=1;
                     {If 4th parameter, use it as first history; else use the one given earlier}
                     IF (ParamCount >= 4) THEN rhnum:=trunc(Params[4]) ELSE rhnum:=Hnum0;
                     {Now accumulate values reading x, y and z components}
                     NEW (Hst2);
                     DI_ReadHist (rhnum, Hst);
                     ApplyQ (Qval,Cval,Lval);
                     FOR t:= 1 TO Tqty2 DO  Hst2^[t]:=sqr(Hst^[t]);
                     DI_ReadHist (rhnum+1, Hst);
                     ApplyQ (Qval,Cval,Lval);
                     FOR t:= 1 TO Tqty2 DO  Hst2^[t]:=Hst2^[t]+sqr(Hst^[t]);
                     DI_ReadHist (rhnum+2, Hst);
                     ApplyQ (Qval,Cval,Lval);
                     FOR t:= 1 TO Tqty2 DO  Hst2^[t]:=Hst2^[t]+sqr(Hst^[t]);
                     FOR t:= 1 TO Tqty2 DO  Hst^[t]:=sqrt(Hst2^[t]);
                     DISPOSE (Hst2);
                   END; {'Q' Vabs of 3 waveforms with Q applied}

             'V','v':  {Calc Vabs - assume given hist is xvel, next yvel, next zvel}
                   BEGIN
                     {If parameter use it as first history, else use the one given earlier}
                     IF (ParamCount >= 1) THEN rhnum:=trunc(Params[1]) ELSE rhnum:=Hnum0;
                     {Now accumulate values reading x, y and z components}
                     NEW (Hst2);
                     DI_ReadHist (rhnum, Hst);
                     FOR t:= 1 TO Tqty2 DO  Hst2^[t]:=sqr(Hst^[t]);
                     DI_ReadHist (rhnum+1, Hst);
                     FOR t:= 1 TO Tqty2 DO  Hst2^[t]:=Hst2^[t]+sqr(Hst^[t]);
                     DI_ReadHist (rhnum+2, Hst);
                     FOR t:= 1 TO Tqty2 DO  Hst2^[t]:=Hst2^[t]+sqr(Hst^[t]);
                     FOR t:= 1 TO Tqty2 DO  Hst^[t]:=sqrt(Hst2^[t]);
                     DISPOSE (Hst2);
                   END; {'V','v' calc Vabs}

             'r','R' :  {Rotate ... to source direction: assume first}
                   BEGIN
                     IF (ParamCount >= 1) THEN rcomp:=trunc(Params[1]) ELSE rcomp:=0; {component, 0:x,1:y,2:k}
                     IF (ParamCount >= 2) THEN rhnum:=trunc(Params[2]) ELSE rhnum:=Hnum0; {history - assumed x, then y and z!}
                     IF (ParamCount >= 3) THEN ri:=trunc(Params[3])  ELSE ri:=0; {ri,rj,rk - SX centre}
                     IF (ParamCount >= 4) THEN rj:=trunc(Params[4])  ELSE rj:=0;
                     IF (ParamCount >= 5) THEN rk:=trunc(Params[5])  ELSE rk:=0;
                     IF (rhnum=0) THEN rhnum:=Hnum0; {starts with original hist number}
                     Hmap := HistMap^[rhnum]; {reset to the given history}
                     r_gam  := arctan0 ((Hmap.k1-rk),(Hmap.i1-ri));
                     r_thet := arctan0 ((Hmap.j1-rj), sqrt( (Hmap.i1-ri)*(Hmap.i1-ri) + (Hmap.k1-rk)*(Hmap.k1-rk) ) );
                     r_cthet := cos(r_thet); r_sthet := sin(r_thet);
                     r_cgam := cos(r_gam); r_sgam := sin(r_gam);

                     {Now accumulate values - read x, y and z components and weight accordingly}
                     FOR t := 1 TO Tqty2  DO Hst^[t]:=0.0;
                     NEW (Hst2);
                     DI_ReadHist ( rhnum, Hst2);
                     FOR t := 1 TO Tqty2  DO
                       CASE rcomp OF
                         {0: Hst^[t] := Hst^[t] + Hst2^[t]*r_cthet*r_cgam;}
                         0: Hst^[t] := Hst^[t] + Hst2^[t]*r_cgam*r_cthet;
                         1: Hst^[t] := Hst^[t] - Hst2^[t]*r_cgam*r_sthet;
                         2: Hst^[t] := Hst^[t] - Hst2^[t]*r_sgam;
                       END; {Case rcomp}
                     DI_ReadHist ( rhnum+1, Hst2);
                     FOR t := 1 TO Tqty2  DO
                       CASE rcomp OF
                         {0: Hst^[t] := Hst^[t] + Hst2^[t]*r_sthet;}
                         0: Hst^[t] := Hst^[t] + Hst2^[t]*r_sthet;
                         1: Hst^[t] := Hst^[t] + Hst2^[t]*r_cthet;
                         2: Hst^[t] := Hst^[t];
                       END; {Case rcomp}
                     DI_ReadHist ( rhnum+2, Hst2);
                     FOR t := 1 TO Tqty2  DO
                       CASE rcomp OF
                         {0: Hst^[t] := Hst^[t] + Hst2^[t]*r_cthet*r_sgam;}
                         0: Hst^[t] := Hst^[t] + Hst2^[t]*r_sgam*r_cthet;
                         1: Hst^[t] := Hst^[t] - Hst2^[t]*r_sgam*r_sthet;
                         2: Hst^[t] := Hst^[t] + Hst2^[t]*r_cgam;
                       END; {Case rcomp}

                     DISPOSE (Hst2);
                   END; {'R' Rotate to source}

             {Tp calculation - "predominant period"}
             't' : BEGIN
                     alpha  := 0.95;
                     {Dsmall := 1e-4;}
                     Dsmall := 1e-20; {mustn't be 0 ... problem in synthetics}
                     NEW (Hst2);
                     IF (ParamCount >= 1) THEN typ:=trunc(Params[1]) ELSE typ:=0;
                     IF (typ > 2) THEN typ:=0;
                     IF (ParamCount >= 2) THEN Ds:=Params[2] ELSE Ds:=100;
                     IF (ParamCount >= 3) THEN asecs:=Params[3] ELSE asecs:=10;
                     IF (asecs <= 0) THEN asecs:=10;
                     alpha := exp (ln(0.1) / (asecs/Hmap.Dt));
                     IF (ParamCount >= 4) THEN rtyp:=trunc(Params[4]) ELSE rtyp:=0;

                     CASE typ OF
                       0: BEGIN {Nakamura version}
                            {Hst = velocity, Hst2=derivative}
                            FOR t := 2 TO (Tqty2-1)  DO
                                Hst2^[t] := (Hst^[t+1] - Hst^[t-1]) / Hmap.Dt * 0.5;
                            Hst2^[1] := (Hst^[2] - Hst^[1]) / Hmap.Dt;
                            Hst2^[Tqty2] := (Hst^[Tqty2] - Hst^[Tqty2-1]) / Hmap.Dt;
                            Nns:=50;
                            IF (Nns > (Tqty2 DIV 10))
                               THEN Nns:=Tqty2 DIV 10;
                            IF (Nns < 2) THEN Nns:=2;
                            FOR t := 1 TO Nns DO
                               {Dsmall:=Dsmall+Hst^[t]*Hst^[t]+Hst2^[t]*Hst2^[t];}
                               Dsmall:=Dsmall+Hst^[t]*Hst^[t];
                            Dsmall:=Dsmall/Nns;  {avg noise}
                            write(tmpdebug, Dsmall);
                            Dsmall:=Dsmall*asecs/Hmap.Dt; {avg * n samples}
                            write(tmpdebug, Dsmall);

                            {Dsmall:=Dsmall/Hmap.Dt/Hmap.Dt;}

                            Dsmall := Dsmall * 40 /Ds/Ds; {2*pi * 2*pi ~ 40}

                            {21/6/2011: ASSUME: if Ds parameter is given, use it directly}

                            writeln(tmpdebug, Dsmall); Flush(tmpdebug);
                            {Dsmall:=Ds*Dsmall;}
                            {IF (Dsmall>10000) THEN Dsmall:=1e9;}
                            IF (ParamCount >= 2) THEN Dsmall:=Ds;
                          END; {Case Tp 0}

                       1: BEGIN {Kanamori version}
                            {Hst = displacement, Hst2=derivative}
                            alpha := 1.0;
                            FOR t := 1 TO (Tqty2) DO
                              Hst2^[t] := Hst^[t];
                            Hst^[1] := 0.5*(Hst2^[2] + Hst2^[1]) * Hmap.Dt;
                            FOR t := 2 TO (Tqty2-1)  DO
                                Hst^[t] := Hst^[t-1] + 0.5*(Hst2^[t+1] + Hst2^[t-1]) * Hmap.Dt;
                            Hst^[Tqty2] := Hst^[Tqty2-1] + 0.5*(Hst2^[Tqty2] + Hst2^[Tqty2-1]) * Hmap.Dt;
                          END; {Case Tp 1}

                       2: BEGIN {acceleration version}
                            {Hst = accel, Hst2=derivative}
                            FOR t := 1 TO (Tqty2) DO
                              Hst2^[t] := Hst^[t];
                            FOR t := 2 TO (Tqty2-1)  DO
                                Hst^[t] := (Hst2^[t+1] - Hst2^[t-1]) / Hmap.Dt * 0.5;
                            Hst^[1] := (Hst2^[2] - Hst2^[1]) / Hmap.Dt;
                            Hst^[Tqty2] := (Hst2^[Tqty2] - Hst2^[Tqty2-1]) / Hmap.Dt;
                            FOR t := 2 TO (Tqty2-1)  DO
                                Hst2^[t] := (Hst^[t+1] - Hst^[t-1]) / Hmap.Dt * 0.5;
                            Hst2^[1] := (Hst^[2] - Hst^[1]) / Hmap.Dt;
                            Hst2^[Tqty2] := (Hst^[Tqty2] - Hst^[Tqty2-1]) / Hmap.Dt;
                          END; {Case Tp 2}

                       ELSE ;
                      END; {Case}

                     {Now accumulate values}
                     Xi:=0.0; Di:=0.0;
                     FOR t := 1 TO Tqty2  DO
                       BEGIN
                         Xi := alpha*Xi + Hst^[t]*Hst^[t];
                         Di := alpha*Di + Hst2^[t]*Hst2^[t];
                         Ti := 2.0*PI*sqrt(Xi/(Di+Dsmall));
                         CASE rtyp OF
                           0: Hst^[t] := Ti;
                           1: Hst^[t] := sqrt(Xi);
                           2: Hst^[t] := sqrt(Di);
                           ELSE Hst^[t] := Ti;
                         END; {Case rtyp}
                       END; {For}
                     DISPOSE (Hst2);
                   END; {'t': Tp option}

             {Tp calculation - Experimental version - "predominant period"}
             {Experimental version - try adding terms}
             'T' : BEGIN
                     alpha  := 0.95;
                     {Dsmall := 1e-4;}
                     Dsmall := 1e-20; {mustn't be 0 ... problem in synthetics}
                     NEW (Hst2);
                     IF (ParamCount >= 1) THEN typ:=trunc(Params[1]) ELSE typ:=0;
                     IF (typ > 2) THEN typ:=0;
                     IF (ParamCount >= 2) THEN Ds:=Params[2] ELSE Ds:=100;
                     IF (ParamCount >= 3) THEN asecs:=Params[3] ELSE asecs:=10;
                     IF (asecs <= 0) THEN asecs:=10;
                     alpha := exp (ln(0.1) / (asecs/Hmap.Dt));
                     IF (ParamCount >= 4) THEN rtyp:=trunc(Params[4]) ELSE rtyp:=0;

                     CASE typ OF
                       0: BEGIN {Nakamura version}
                            {Hst = velocity, Hst2=derivative}
                            FOR t := 2 TO (Tqty2-1)  DO
                                Hst2^[t] := (Hst^[t+1] - Hst^[t-1]) / Hmap.Dt * 0.5;
                            Hst2^[1] := (Hst^[2] - Hst^[1]) / Hmap.Dt;
                            Hst2^[Tqty2] := (Hst^[Tqty2] - Hst^[Tqty2-1]) / Hmap.Dt;
                            Nns:=50;
                            IF (Nns > (Tqty2 DIV 10))
                               THEN Nns:=Tqty2 DIV 10;
                            IF (Nns < 2) THEN Nns:=2;
                            FOR t := 1 TO Nns DO
                               {Dsmall:=Dsmall+Hst^[t]*Hst^[t]+Hst2^[t]*Hst2^[t];}
                               Dsmall:=Dsmall+Hst^[t]*Hst^[t];
                            Dsmall:=Dsmall/Nns;  {avg noise}
                            write(tmpdebug, Dsmall);
                            Dsmall:=Dsmall*asecs/Hmap.Dt; {avg * n samples}
                            write(tmpdebug, Dsmall);

                            {Dsmall:=Dsmall/Hmap.Dt/Hmap.Dt;}

                            Dsmall := Dsmall * 40 /Ds/Ds; {2*pi * 2*pi ~ 40}

                            writeln(tmpdebug, Dsmall); Flush(tmpdebug);
                            {Dsmall:=Ds*Dsmall;}
                            {IF (Dsmall>10000) THEN Dsmall:=1e9;}
                          END; {Case Tp 0}

                       1: BEGIN {Kanamori version}
                            {Hst = displacement, Hst2=derivative}
                            alpha := 1.0;
                            FOR t := 1 TO (Tqty2) DO
                              Hst2^[t] := Hst^[t];
                            Hst^[1] := 0.5*(Hst2^[2] + Hst2^[1]) * Hmap.Dt;
                            FOR t := 2 TO (Tqty2-1)  DO
                                Hst^[t] := Hst^[t-1] + 0.5*(Hst2^[t+1] + Hst2^[t-1]) * Hmap.Dt;
                            Hst^[Tqty2] := Hst^[Tqty2-1] + 0.5*(Hst2^[Tqty2] + Hst2^[Tqty2-1]) * Hmap.Dt;
                          END; {Case Tp 1}

                       2: BEGIN {acceleration version}
                            {Hst = accel, Hst2=derivative}
                            FOR t := 1 TO (Tqty2) DO
                              Hst2^[t] := Hst^[t];
                            FOR t := 2 TO (Tqty2-1)  DO
                                Hst^[t] := (Hst2^[t+1] - Hst2^[t-1]) / Hmap.Dt * 0.5;
                            Hst^[1] := (Hst2^[2] - Hst2^[1]) / Hmap.Dt;
                            Hst^[Tqty2] := (Hst2^[Tqty2] - Hst2^[Tqty2-1]) / Hmap.Dt;
                            FOR t := 2 TO (Tqty2-1)  DO
                                Hst2^[t] := (Hst^[t+1] - Hst^[t-1]) / Hmap.Dt * 0.5;
                            Hst2^[1] := (Hst^[2] - Hst^[1]) / Hmap.Dt;
                            Hst2^[Tqty2] := (Hst^[Tqty2] - Hst^[Tqty2-1]) / Hmap.Dt;
                          END; {Case Tp 2}

                       ELSE ;
                      END; {Case}

                     {Now accumulate values}
                     afact := 0.5*ln(0.1)/asecs * 1;
                     Xi:=0.0; Di:=0.0;
                     FOR t := 1 TO Tqty2  DO
                       BEGIN
                         Xi := alpha*Xi + Hst^[t]*Hst^[t];
                         Di := alpha*Di + (Hst2^[t] - afact*Hst^[t]) * (Hst2^[t] - afact*Hst^[t]);
                         Ti := 2.0*PI*sqrt(Xi/(Di+Dsmall));
                         CASE rtyp OF
                           0: Hst^[t] := Ti;
                           1: Hst^[t] := sqrt(Xi);
                           2: Hst^[t] := sqrt(Di);
                           ELSE Hst^[t] := Ti;
                         END; {Case rtyp}
                       END; {For}
                     DISPOSE (Hst2);
                   END; {'T': Tp option}

           END; {Case}
         END; {Func1b}



       BEGIN {Parse}
         {Still can use -  j,k,n,o,q,u,y }
         WHILE (Length (Pstring) <> 0) DO
           CASE Pstring[1] OF
             '('      : ParseBrackets(Hst);
             '['      : ParseHnum;
             'I', 'i', 'D', 'd', 'F', 'f', 'P', 'p', 'T', 't',
             'M', 'm', 'S', 's', 'X', 'x', 'E', 'e', '#', 'L', 'l', 'J', 'j' :
                        Func1 (Pstring[1]);
             '+', '-', '*', '/', 'C', 'c', 'R', 'r' :
                        Func2 (Pstring[1]);
             'G', 'g', 'V','v', 'W','w', 'Z','z', 'A', 'a','B', 'b' :
                        Func2b (Pstring[1]);
{             'G', 'g','V', 'v' : Func2b (Pstring[1]);}

             '&' : BEGIN { & Extended functions - of 1 variable }
                     Delete (Pstring, 1, 1);
                     IF (Pstring[1] IN ['F','f','X','x','D','d','R','r','W','w',
                                        'p','P','Q','q','V','v','T','t'])
                        THEN Func1B(Pstring[1]) ELSE Delete(Pstring,1,1);
                   END;
             '$' : BEGIN { $ Extended functions - of 1 variable}
                     Delete (Pstring, 1, 1);
                     IF (Pstring[1] IN ['G','g','H','h','*'])
                        THEN Func2C(Pstring[1]) ELSE Delete(Pstring,1,1);
                   END;

             ' ' : Delete (Pstring, 1, 1);
             ELSE  Delete (Pstring, 1, 1);
           END; {While / Case}
       END; {Parse}

    BEGIN
    END. {DspForm}
