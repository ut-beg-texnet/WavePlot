PROGRAM WavePlot;
  { Receives an output file from WAVE & allows the user to view graphs
      on screen and output to plotter.}

{$M $A100,0,$A0000}   {Stack-size, heapmin, heapmax}
{$G+}                 {80286 instruction set}
{$N+}                 {Use numeric co-processor}
{$F+}                 {Force Far Calls}

  USES
    Crt, Graph, Funct, Setts, Data_rsc, Plprim, Dsprim, Dspl, Plot{, Menu_Rsc};

  TYPE
    ShowTypes = (Snaps, Dumps, Hists);
  VAR
{aa,bb,cc,dd,  r1,r2,r3, r1i,r2i,r3i : single; {temp}
    Choice, Ch1  : Char;
    ImageNum     : Word;
    ImageStr     : String;
    FileStr      : String;
    ShowType     : ShowTypes;
    i            : Word;
    TrajOn       : 0..2;
    CrackT       : 0..3;
    SnapStart, DumpStart, HistStart, GeomStart  : Word;
    Help_context : HelpType;

  {* * * * * * * * * * * * * * * * * * * * * * * * * * * *}

  PROCEDURE SetupGraph;
    BEGIN {Set up graphics & windows}
      Ds_InitVid;
      Set_colours;
      Ds_InitWindows;
{      SetGraphMode (GraphMode);}
      Ds_256Palette;
    END;  {SetupGraph}

  PROCEDURE SetGscreen;
    BEGIN {Set up graphics & windows}
      ResetViewPort;
      ClearViewPort;
      SetBkColor (DsCol_Bgrnd);
      ClearViewPort;
      SetColor (DsCol_Fgrnd);
      Ds_DrawWindow (MsgWind);
      Ds_DrawWindow (LegWind);
{      Menu_Init;}
      ResetViewPort;
    END; {SetGscreen}

  PROCEDURE SetIscreen;
    BEGIN {Set up an info window}
      IF (sets.colscale = col_blackb)
         THEN SetBkColor (Blue);
      ResetViewPort;
      SetColor (yellow);
      TextColor (white);
      ClearViewPort;
      SetTextStyle (SmallFont, HorizDir, FontSz);
      SetTextJustify (LeftText, TopText);
      wr_clear;
    END; {SetIscreen}


  PROCEDURE SnapSummary;
    VAR
      i       : Integer;
      ListOn  : Boolean;
    BEGIN {SnapSummary}
      ListOn := True;
      FOR i := 1 TO QtySnap DO
        BEGIN
          IF ( ( (i-1) MOD 18) = 0)
             THEN BEGIN
                    IF (ListOn AND (i > 1))
                       THEN BEGIN
                              Writeln (wr_file);
                              Writeln (wr_file, '    ....... Any Key to Continue');
                              wr_flush(wr_hold);
                              IF (ReadKey = #27) THEN ListOn := False;
                            END;
                    wr_flush(wr_nohold);
                    Writeln (wr_file, 'SUMMARY OF PLOTS AVAILABLE'); Writeln(wr_file);
                    Writeln (wr_file,' Snap     Time      Var       Xqty  Yqty         Max         Min     Geom');
                    Writeln (wr_file);
                  END;
          WITH SnapMap^[i] DO
            Writeln (wr_file, i:4, Strng(Time,$F302):12, VarName(SnapVar):8,
                              VFuncName[Sfunc]:4, Xqty:6, Yqty:6,
                              Strng(MaxVal,$F302):14, Strng(MinVal,$F302):12, gnum:7);
{                              rstrng(i1,i2):12, rstrng(j1,j2):12, rstrng(k1,k2):12,}
        END; {For}
      Writeln(wr_file);
      Write (wr_file,'   Enter Snapshot number with which to start: ');
      wr_flush(wr_hold);
      ReadLn (SnapStart);
    END; {SnapSummary}


  PROCEDURE DumpSummary;
    VAR
      i       : Integer;
      ListOn  : Boolean;
    BEGIN {ShowSummary}
      ListOn := True;
      FOR i := 1 TO QtyDump DO
        BEGIN
          IF ( ( (i-1) MOD 18) = 0)
             THEN BEGIN
                    IF (ListOn AND (i > 1))
                       THEN BEGIN
                              Writeln (wr_file);
                              Writeln (wr_file, '    ....... Any Key to Continue');
                              wr_flush(wr_hold);
                              IF (ReadKey = #27) THEN ListOn := False;
                            END;
                    wr_flush(wr_nohold);
                    Writeln (wr_file, 'SUMMARY OF PLOTS AVAILABLE'); Writeln(wr_file);
                    Writeln (wr_file,' Dump      Time          i           j          k      Vqty   Geom');
                    Writeln (wr_file);
                  END;
          WITH DumpMap^[i] DO
            Writeln (wr_file, i:5, Strng(Time,$F302):12, rstrng(i1,i2):12,
                                   rstrng(j1,j2):12, rstrng(k1,k2):12, Vqty:7, gnum:7);
        END; {For}
      Writeln(wr_file);
      Write (wr_file,'   Enter Dumpshot number with which to start: ');
      wr_flush(wr_nohold);
      ReadLn (DumpStart);
    END; {DumpSummary}


  PROCEDURE HistSummary;
    VAR
      i           : Integer;
      rMaxVal     : Real;
      ListOn, Quit  : Boolean;
      FileExists    : Boolean;
      ff            : Text;
    BEGIN {ShowSummary}
      ListOn := True;
      FOR i := 1 TO QtyHist DO
        BEGIN
          IF ( ListOn AND (((i-1) MOD 18)=0) AND (i<QtyHist-2) )
             THEN BEGIN
                    IF (i > 1)
                       THEN BEGIN
                              Writeln (wr_file);
                              Writeln (wr_file, '    ....... Any Key to Continue');
                              wr_flush(wr_hold);
                              IF (ReadKey = #27) THEN ListOn := False;
                            END;
                    wr_flush(wr_nohold);
                    Writeln (wr_file, 'SUMMARY OF PLOTS AVAILABLE'); Writeln(wr_file);
                    Writeln (wr_file,' Hist Steps     T-st    T-fin    Var     X     Y     Z       Max      Min');
                    Writeln (wr_file);
                  END;
          WITH HistMap^[i] DO
            Writeln (wr_file, i:5, Tqty:6, Strng (Tstart,$F202):9, Strng (Time,$F202):9,
                              VarName(HistVar):7, VFuncName[Hfunc]:2,
                              Xp:6, Yp:6, Zp:6, Strng (MaxVal,$F302):10, Strng (MinVal,$F302):10 );
        END; {For}

      Writeln(wr_file);
      Writeln (wr_file,'   Enter Histories to display (max ', MaxSeism, ', 0 to end) ');
      wr_flush(wr_hold);
      QtySeism := 0;
      REPEAT
        Quit := False; Inc (QtySeism);
        WITH Seism[QtySeism] DO
          BEGIN
            Hnum := QtyHist + 1;
            WHILE (Hnum > QtyHist) DO
              Read (Hnum);
            IF (Hnum = 0)
               THEN BEGIN Quit := True; Dec(QtySeism); END
               ELSE BEGIN
                      Hstmap  := HistMap^[Hnum];
                      Formset := False;
                      IF (HstMap.Ax1 IN ['T','t'])
                         THEN GR.xstr := 's'
                         ELSE GR.xstr := 'm';
                      GR.ScalT := LinScal;
                    END;
          END; {With}
      UNTIL ( (QtySeism>=MaxSeism) OR (Quit AND (QtySeism>=1))  );
      ReadLn;

      wr_clear;
      Writeln (wr_file);
      Writeln (wr_file,'           *** FORMULAE ENTRY MODE ***');
      WriteLn (wr_file,'If required, Enter Formulae to create constructed/ calculated');
      writeln (wr_file,'             data sets for any of the above graph numbers.');
      WriteLn (wr_file,'   Entry Format :  Enter Graph number, <space>, formula <CR> .');
      WriteLn (wr_file,'   To end formula entry mode, enter  "0 <CR>", on its own line.');
      Writeln (wr_file,'           *** NB: check syntax ***');
      Writeln (wr_file);
      REPEAT
        Writeln (wr_file);
        WriteLn (wr_file,'Formulae:  Enter Graph number, <space>, formula <CR> - or 0 <CR> to stop.');
        Writeln (wr_file,'   I() integrate, D() differentiate, F() Fourier Transform (Mag)');
        Writeln (wr_file,'   P() Fourier Transform (Phase)   , t() filtered time response');
        Writeln (wr_file,'   "+" add, "-" subtract, "*" multiply, [num] history number');
        Writeln (wr_file,'   *** For others and syntax, see manual ***');
        wr_flush(wr_nohold);
        Read (i);
        IF ( (i > 0) AND (i <= QtySeism) )
           THEN BEGIN
                  Readln (Seism[i].Form);
                  IF (Seism[i].Form<>'')
                     THEN Seism[i].FormSet := True
                     ELSE
                      BEGIN
                       {$I-}
                       WITH Sets DO
                         ASSIGN (ff,(Drive+':'+Dir+'\'+ 'form.'+strng(i,$00)) );
                       Reset(ff);
                       FileExists := (IOResult=0);
                       {$I+}
                       IF (FileExists)
                          THEN BEGIN Readln(ff,Seism[i].Form); close(ff); Seism[i].FormSet := True; END;
                      END;

{                  IF (FileExists)
                     THEN BEGIN Readln(ff,Seism[i].Form); close(ff); END
                     ELSE Readln (Seism[i].Form);}
                  {Seism[i].FormSet := True;}
  {                IF (LENGTH (Seism[i].Form) = 2)
                     THEN  Seism[i].Form := Seism[i].Form + '([' + STRNG(Seism[i].Hnum, $00) + '])';}
                END;
      UNTIL (i = 0);
      ReadLn;
    END; {HistSummary}


  PROCEDURE GraphSingleSnap;
    VAR
      Count, G  : Word;
    BEGIN
      Ds_SnapScale (CurSnap);
      Ds_ClearWindow (GraphWind);
      Ds_Graph2dAxes;

{     Read geometry. First identify geometry corresponding to snapshot gID}
      G := 0;
      IF (Scale2D.TrueGscale) THEN
        WITH SnapMap^ [CurSnap] DO
          DI_ReadGeom (G, Gnum, gID);

      IF ((Scale2D.TrueGscale) AND NOT(Keypressed))  THEN Ds_Graph2DGeom (G,Slice2D);
      DI_ReadSnap (CurSnap);
      Ds_GraphSnap;
      Ds_ClearWindow (LegWind);
      Ds_ListLevels;
      IF ((Scale2D.TrueGscale) AND NOT(Keypressed))  THEN Ds_Graph2DGeom (G,Slice2D);
      Draw_HistPos;
      DI_ClearGeom (G);
    END; {GraphSingleSnap}


  PROCEDURE GraphMultiSnap;
    VAR
      Count, G    : Word;
      Fnamebk     : FileName;
      SnapWindBk  : Ds_Window;
      CurSnapBk   : Word;
      Hz, Vt      : Word;
      whstp, wvstp, whsz, wvsz, whst, wvst : Word;
      Vsz, Voff, Hsz, Hoff                 : Integer;
      MMaxV, MMinV : single;
    BEGIN
      {Backup variables}
      FnameBk := Copy (Sets.Fname,1,8);
      SnapWindBk := SnapWindMx;
      CurSnapBk  := CurSnap;
      Ds_ClearWindow (GraphWind);
      Ds_ClearWindow (LegWind);
      Ds_SnapScale (CurSnap);

      {calc windows}
      Vsz  := (Vpix-3); Hsz := TRUNC (Vsz/Aspect); Voff := Vsz DIV 15; Hoff := Hsz DIV 25;
      WITH Sets DO
        BEGIN
          whstp := (Hsz-Hoff) DIV whqty;      whsz := TRUNC(0.9*whstp); whst := VtTWind.Fin[1]+1+TRUNC(0.05*whstp);
          wvstp := (Vsz-Voff-Voff) DIV wvqty; wvsz := TRUNC(0.9*wvstp); wvst := HdTWind.Fin[2]+1+TRUNC(0.05*wvstp);
        END;

      {get max and min from each snapshot}
      MMaxV := -LargeSingle;  MMinV :=  LargeSingle;
      WITH Sets DO
        FOR Vt := 0 TO (wvqty-1) DO
         FOR Hz := 0 TO (whqty-1) DO
          BEGIN
            IF (MultFile)
               THEN BEGIN  Fname := Mfiles[Hz,Vt]; DI_Close; DI_AssignFile (Fname); DI_ReadMap;  END;
            IF (SnapMap^[CurSnap].MaxVal>MMaxV) THEN MMaxV := SnapMap^[CurSnap].MaxVal;
            IF (SnapMap^[CurSnap].MinVal<MMinV) THEN MMinV := SnapMap^[CurSnap].MinVal;
            IF NOT(MultFile)
               THEN BEGIN  CurSnap := CurSnap + Mjump; IF (CurSnap > QtySnap) THEN CurSnap := 1;  END;
          END; {For}
      CurSnap    := CurSnapBk;
      Sets.Fname := Fnamebk;

      {draw each snapshot}
      WITH Sets DO
        FOR Vt := 0 TO (wvqty-1) DO
         FOR Hz := 0 TO (whqty-1) DO
          BEGIN
            WITH SnapWindMx DO
              BEGIN
                Sz[1] := whsz;  St[1] := whst + Hz*whstp;  Fin[1] := St[1]+Sz[1]-1;
                Sz[2] := wvsz;  St[2] := wvst + Vt*wvstp;  Fin[2] := St[2]+Sz[2]-1;
              END; {With SnapWind}
            IF (MultFile)
               THEN BEGIN  Fname := Mfiles[Hz,Vt]; DI_Close; DI_AssignFile (Fname); DI_ReadMap;  END;
            DI_ReadSnap (CurSnap);
            {should allow option, whether to link}
            IF (NOT MultNoLink)
               THEN BEGIN CurSnapRec.MaxVal:=MMaxV; CurSnapRec.MinVal:=MMinV; END;
            Ds_SnapScale (CurSnap);
            {Read geometry. First identify geometry corresponding to snapshot gID}
            G := 0;
            IF (Scale2D.TrueGscale) THEN
              WITH SnapMap^ [CurSnap] DO
                DI_ReadGeom (G, Gnum, gID);
            IF ((Scale2D.TrueGscale) AND NOT(Keypressed))  THEN Ds_Graph2DGeom (G,Slice2D);
            Ds_GraphSnap;
            IF ((Scale2D.TrueGscale) AND NOT(Keypressed))  THEN Ds_Graph2DGeom (G,Slice2D);
            IF NOT(MultFile)
               THEN BEGIN  CurSnap := CurSnap + Mjump; IF (CurSnap > QtySnap) THEN CurSnap := 1;  END;
            DI_ClearGeom (G);
          END; {For}

      {restore}
      SnapWindMx := SnapWindBk;
      CurSnap    := CurSnapBk;
      Sets.Fname := Fnamebk;
      IF (Sets.MultFile)
         THEN BEGIN  DI_Close; DI_AssignFile (Sets.Fname); DI_ReadMap;  END;

      CurSnapRec.MaxVal:=MMaxV;
      CurSnapRec.MinVal:=MMinV;
      Ds_ListLevels;
    END; {GraphMultiSnap}


  PROCEDURE GraphDump;
    VAR
      Count, G    : Word;
    BEGIN
      Ds_ClearWindow (GraphWind);
      Ds_ClearWindow (LegWind);
      Ds_DumpScale (CurDump);
      Ds_Graph2dAxes;
      {Read geometry. First identify geometry corresponding to dump gID}
      G := 0;
      WITH DumpMap^ [CurDump] DO
        DI_ReadGeom (G, Gnum, gID);
      IF NOT(Keypressed)  THEN Ds_Graph2DGeom (G,Slice2D);
      IF NOT(Keypressed)  THEN Ds_GraphDump;
      IF NOT(Keypressed)  THEN Ds_Graph2DGeom (G,Slice2D);
      DI_ClearGeom (G);
    END; {GraphDump}


  PROCEDURE Common2DChoice (Choice:Char);
      { Deals with the following common requests:
          #12, #19,'Z','z','L','l','G','g','E','e',
          #07,'A','a','I','i','F','f','D','d','>','<',
          'X','x','C','c','H','h','V','v','B','b',
          '+','-','T','t',
          #0 }
    VAR
      Ch1 : Char;
    BEGIN
      WITH Sets DO
        CASE Choice OF
         {Special codes}
          #0       : BEGIN
                       Ch1 := Ds_ReadKey;
                       CASE Ch1 OF
(*                         #68,#33,#34,#48,#47,
                         #25,#46  : RunMenu (Ch1);                   {'F10',Alt-F,G,B,V,P,C - Menus}
                         -- Not actually available anymore!!   *)
                         #59      : Ds_ShowHelp(Help_context);       {'F1' Help}

                         #77      : Ds_Shift ( 1, 0);                {Right Cursor}
                         #75      : Ds_Shift (-1, 0);                {Left Cursor}
                         #72      : Ds_Shift (0 , 1);                {Up Cursor}
                         #80      : Ds_Shift (0 ,-1);                {Down Cursor}

                         #116     : Ds_Enlarge ( 1,  0);              {Ctrl-Right Cursor}
                         #115     : Ds_Enlarge (-1,  0);              {Ctrl-Left Cursor}
                         #132     : Ds_Enlarge (0  , 1);              {Ctrl-PageUp Cursor}
                         #118     : Ds_Enlarge (0  ,-1);              {Ctrl-PageDown Cursor}

(*                         #116     : Ds_Shift ( 10,  0);              {Ctrl-Right Cursor}
                         #115     : Ds_Shift (-10,  0);              {Ctrl-Left Cursor}
                         #132     : Ds_Shift (0  , 10);              {Ctrl-PageUp Cursor}
                         #118     : Ds_Shift (0  ,-10);              {Ctrl-PageDown Cursor}*)

                         #46      : Ds_Zoom (-1);                    {Alt-c}
                         #45      : Ds_Zoom (1);                     {Alt-x}
                         #20      : ShowTitles := NOT(ShowTitles);   {Alt-T:  Titles}
                         #50      : MatOn := (MatOn+1) MOD 4;        {Alt-m}

                         #83      : ang_y := (ang_y+2) MOD 360; {Del}
                         #81      : ang_y := (ang_y-2) MOD 360; {PageDn}
                         #71      : ang_x := (ang_x-2) MOD 360; {Home}
                         #79      : ang_x := (ang_x+2) MOD 360; {End}

                         #34      : BEGIN                            {Alt-G - graphics colours}
                                      IF (sets.colscale=Col_Grey)
                                         THEN sets.colscale:=Col_Blackb
                                         ELSE sets.colscale:=succ(sets.colscale);
                                      IF (Rainbow)
                                         THEN Ds_SetPalette
                                         ELSE Ds_ResetPalette;
                                    END;
                       END; {Case}
                     END; {Special codes}

         {Geometry}
          'Z'      : Ds_Zoom (1);                            {zoom}
          'z'      : Ds_Zoom (-1);                           {unzoom}
          #12      : GeomExtSet := NOT(GeomExtSet);          {Ctrl-L:  Link/Unlink geometric extents}
          #19      : IF (GeomScSet)                          {Ctrl-S:  Link/Unlink geometric scales}
                        THEN GeomScSet := False
                        ELSE IF (GeomSc <> 0)
                                THEN GeomScSet := True;
          'l'      : VectSet := False;
          'L'      : BEGIN
                       IF (VectSc = 0) THEN VectSc:=1;
                       VectSet := True;                      {Link Scales with prev. snapshots}
                     END;

         {Graph Retrieval}
          'G'      : MapOn   := True;                        {Values mapped and averaged in Fast mode}
          'g'      : MapOn   := False;
          'E'      : QuickMax := iMin ( QuickMax*2 - 1,  MaxX);
          'e'      : QuickMax := iMax ( (QuickMax DIV 2) + 1,3);

         {Snapshot type graphs}
          #07      : IF (Sets.SnapSet=Mline )                {Ctrl-G}
                        THEN Sets.SnapSet := ColMap
                        ELSE Sets.SnapSet := Succ (Sets.SnapSet);
          'A'      : SnapColAbs := True;                     {Absolute vals for colours}
          'a'      : SnapColAbs := False;
          'I'      : Interp  := True;                        {Interpolate}
          'i'      : Interp  := False;
          'F'      : Quick   := True;                        {Fast mode}
          'f'      : Quick   := False;
          'D', 'd' : IF (SnapFill = Lin)                     {Distribution type for colour scale}
                        THEN SnapFill := Log
                        ELSE SnapFill := succ(SnapFill);
          '>'      : Inc (QtyLevels);                        {Increase no. of levels}
          '<'      : Dec (QtyLevels);

         {Vector and Multi-line graphs}
          'X'      : Vzoom := Vzoom * 1.25;                  {eXpand vector scale}
          'x'      : Vzoom := Vzoom * 0.8;
          'C','c'  : LineCol := NOT (LineCol);               {Toggle Colour mode}
          'H', 'h' : Dirn    := Horiz;                       {Horizontal}
          'V', 'v' : Dirn    := Vert;                        {Vertical}
          'B', 'b' : Dirn    := Both;                        {Horizontal and Vertical lines}

         {Miscellaneous}
          '+'      : Tdelay := TRUNC (Tdelay * 0.8);         {speed up}
          '-'      : Tdelay := TRUNC (Tdelay * 1.25 + 1);    {slow down}
          't'      : Ds_ResetPalette;
          'T'      : Ds_SetPalette;

        END; {Case and with Sets}
    END; {Common2DChoice}


  PROCEDURE ShowSnaps;
    CONST
      Ln2 = 0.6931471806;
    VAR
      Choice, Ch : Char;
      s          : Word;
      Sstr        : String;
      Num, Err, Jump : Integer;
      Wait, AutoSave : Boolean;
    BEGIN
      SetIscreen;
      Help_context := HelpSnap;

      SnapSummary;
      Ds_SetPalette;
      SetGscreen;
      NEW  (SnapShot);
      Vzoom := LineGzoom; Scale2D.Magn  := 0; Jump := 1; CurSnap := SnapStart;
      MultNoLink := False;
      REPEAT
        Ds_ClearWindow (MsgWind);
        Ds_AddMessage (1,4, '<M>ap colours' );
        Ds_AddMessage (2,4, '<C>ontours');
        Ds_AddMessage (3,4, '<L>ine Repr.');
        Ds_AddMessage (4,4, '<Q>uit');
        Ds_WaitResponse (Choice);
        CASE Choice OF
          'M', 'm' : Sets.SnapSet := ColMap;
          'L', 'l' : Sets.SnapSet := MLine;
          'C', 'c' : Sets.SnapSet := ContMap;
        END; {Case}
        Sets.VectSet := False;

        Wait := True; AutoSave := False;
        Ch := Choice;
        WHILE ( NOT (Choice IN ['Q', 'q', #27]) ) DO
          BEGIN
            IF (Sets.MultSnap)
               THEN GraphMultiSnap
               ELSE GraphSingleSnap;
            Ds_ClearWindow (MsgWind);
            Ds_AddMessage (1,6, '<S>ave  <P>lot' );
            Ds_AddMessage (2,6, '<M>ode  <Q>uit');
            Ds_AddMessage (3,6, '(0..9) graph num.');
            CASE Sets.SnapSet OF
               ColMap  : Ds_AddMessage (4,6, '<I>nterp <F>ast');
               ContMap : Ds_AddMessage (4,6, 'Contour: ');
               MLine   : Ds_AddMessage (4,6, 'Lines: X,C,H,V,B');
            END; {Case}
            Ds_AddMessage (5,6, 'Scale: A,L,K,R,>,<');
            Ds_AddMessage (6,6, '  ... or next page');

            IF (Wait)
               THEN Choice := Ds_ReadKey
               ELSE IF (KeyPressed)
                       THEN Choice := Ds_ReadKey
                       ELSE IF (AutoSave)
                               THEN Choice := 's'
                               ELSE BEGIN  Choice := ' '; Delay(600);  END;
            WITH Sets DO
              CASE Choice OF
                {Common requests: #0,#07,#12,#19}
                {   A,a,B,b,C,c,D,d,E,e,F,f,G,g,H,h,I,i,L,l,T,t,V,v,X,x}
                {Other requests:  J,j,K,k,M,m,N,n,O,o,P,p,R,r,S,s,U,u,Y}
                #12,#19,'Z','z','L','l','G','g','E','e',
                #07,'A','a','I','i','F','f','D','d','>','<',
                'X','x','C','c','H','h','V','v','B','b',
                '+','-','T','t',
                #0
                         : Common2DChoice(Choice);
                'U'      :  DI_CvertAscii2(0);  {Generate ascii file of max line, from line hist}
                'u'      :  DI_CvertAscii2(1);  {Generate ascii file of min line, from line hist}
                'Y'      :  DI_CvertAscii2(2);  {Generate ascii file of ABS MAX line, from line hist}

                'N', 'n' : Wait:=NOT(Wait);   {No wait}
                'O', 'o' : AutoSave:=NOT(AutoSave);   {Auto Save}
                'S', 's' : BEGIN {Save}
                             Ds_ClearWindow (MsgWind);
                             str (CurSnap,ImageStr);
                             FileStr  := Sets.Drive+':'+Sets.Dir+'\'+Sets.Fname;
                             Ds_AddMessage (3,6, 'FILE: '+ FileStr);
                             Ds_AddMessage (4,6, 'SNAPSHOT: '+ImageStr);
                             IF (CurSnap > 99)
                                THEN Delete (ImageStr, 1, Length(ImageStr)-2);
                             IF (CurSnap < 10)
                                THEN ImageStr := '0' + ImageStr;
                             Ds_SaveImage (FileStr,ImageStr,(Sets.Drive+':'+Sets.Dir+'\'));
                             CurSnap := CurSnap+Jump;
                           END;
                'P', 'p' : PlotSnap (CurSnap);
                '1'..'9' : BEGIN
                             s := 0;
                             REPEAT
                               s := s*10 + IVal(Choice);
                               Choice := ReadKey;
                             UNTIL NOT (Choice IN ['0'..'9']);
                             Choice := ' ';
                             CurSnap := s;
                           END;
                'M', 'm' : BEGIN {toggle Graphics mode}
                             col256 := NOT (col256);
                             IF (col256 AND (Sets.VidMode IN [std16]))
                                THEN BEGIN
                                       SetIscreen;
                                       Set_VidMode;
                                       CloseGraph;
                                       SetupGraph;
                                       SetGscreen;
                                     END;
                             Set_colours;
                             Ds_SetPalette;
                           END;
(*                'I'      : Interp  := True;      {Interpolate}
                'i'      : Interp  := False;
                'F'      : Quick   := True;      {Fast mode}
                'f'      : Quick   := False;
                'D', 'd' : IF (SnapFill = Lin)     {Distribution type for colour scale}
                              THEN SnapFill := Log
                              ELSE SnapFill := succ(SnapFill);
                'C','c'  : LineCol := NOT (LineCol);  {Toggle Colour mode}
                'H', 'h' : Dirn    := Horiz;     {Horizontal}
                'V', 'v' : Dirn    := Vert;      {Vertical}
                'B', 'b' : Dirn    := Both;      {Horizontal and Vertical lines}
*)                'K', 'k' : BEGIN {set fixed scale, keep according to current scale}
                             IF (Sets.SnapSet = MLine)
                                THEN VectSc  := (Hpix/ScrnLen) / Ds_VectSc;
                             MaxV    := CurSnapRec.MaxVal;
                             MinV    := CurSnapRec.MinVal;
                           END;
                'R'      : BEGIN {double Range for colour level distribution}
                             SnapNLev := Exp ( ROUND ( (Ln(SnapNLev)/Ln(2)) + 1) * Ln2);
                             Snap1Lev := Exp (Ln(SnapNLev) / QtyLevels);
                           END;
                'r'      : BEGIN
                             SnapNLev := Exp ( ROUND ( (Ln(SnapNLev)/Ln(2)) - 1) * Ln2);
                             Snap1Lev := Exp (Ln(SnapNLev) / QtyLevels);
                           END;
                'J','j'  : {Jump length}
                           BEGIN
                             Ds_GetInput ('JUMP FACTOR',Sstr); Val (Sstr, Num, Err);
                             IF ((Num >= 1) AND (Num <= QtySnap))
                                THEN Jump := Num;
                           END;
                #24      : {Ctrl-X: extracts maximum snapshots}
                           DI_ExtractSnap;
                #21      : {Ctrl-U: causes multi-snaps not to be linked}
                           MultNoLink:=NOT(MultNoLink);
                ELSE CurSnap := CurSnap+Jump;
              END; {Case and with Sets}

            Ds_ClearWindow (MsgWind);
            IF (CurSnap > QtySnap)
               THEN CurSnap := 1;
          END; {While}
      UNTIL (Ch IN ['Q', 'q', #27]);
      DISPOSE (SnapShot);
    END; {ShowSnaps}


  PROCEDURE ShowDumps;
    VAR
      Choice, Ch : Char;
      s          : Word;
      Sstr        : String;
      Num, Err, Jump : Integer;
      Wait, AutoSave : Boolean;
    BEGIN
      SetIscreen;
      Help_context := HelpDump;
      DumpSummary;
      Ds_ResetPalette;
      SetGscreen;
      Vzoom := VectZoom; Scale2D.Magn  := 0; Jump := 1; CurDump := DumpStart;
      REPEAT
        Ds_ClearWindow (MsgWind);
        Ds_AddMessage (1,4, '<S>tresses' );
        Ds_AddMessage (2,4, '<V>elocity Vectors');
        Ds_AddMessage (3,4, '<E>SS');
        Ds_AddMessage (4,4, '<I>sochromatics');
        Ds_WaitResponse (Choice);
        CASE Choice OF
          'S', 's' : Sets.DumpSet := Stress;
          'V', 'v' : Sets.DumpSet := Vvect;
          'E', 'e' : Sets.DumpSet := ESS;
          'I', 'i' : Sets.DumpSet := ISO;
        END; {Case}
        CASE Sets.DumpSet OF
          Stress, Vvect : NEW (Dump);
          ESS, ISO      : BEGIN  NEW (SnapShot); Sets.SnapSet := ContMap;  END;
        END; {Case}
        Sets.VectSet := False;
        Wait := True; AutoSave := False;
        Ch := Choice;

        WHILE ( NOT (Choice IN ['Q', 'q', #27]) ) DO
          BEGIN
            GraphDump;
            Ds_ClearWindow (MsgWind);
            Ds_AddMessage (1,6, '<S>ave  <P>lot' );
            Ds_AddMessage (2,6, '<Q>uit');
            Ds_AddMessage (3,6, '(0..9) graph num.');
            Ds_AddMessage (4,6, '<V>ector Type');
            Ds_AddMessage (5,6, 'Scale: A,L,K');
            Ds_AddMessage (6,6, '  ... or next page');

            IF (Wait)
               THEN Choice := Ds_ReadKey
               ELSE IF (KeyPressed)
                       THEN Choice := Ds_ReadKey
                       ELSE IF (AutoSave)
                               THEN Choice := 's'
                               ELSE BEGIN  Choice := ' '; Delay(600);  END;
            WITH Sets DO
              CASE Choice OF
                #12, #19,'Z','z','L','l','G','g','E','e',
                #07,'A','a','I','i','F','f','D','d','>','<',
                'X','x','C','c','H','h','V','v','B','b',
                '+','-','T','t',
                #0
                         : Common2DChoice(Choice);
                'N', 'n' : Wait:=NOT(Wait);   {No wait}
                'S', 's' : BEGIN {Save}
                             Ds_ClearWindow (MsgWind);
                             str (CurDump,ImageStr);
                             ImageStr := Strng(CurDump, 0);
                             FileStr  := Sets.Drive+':'+Sets.Dir+'\'+Sets.Fname;
                             Ds_AddMessage (3,6, 'FILE: '+ FileStr);
                             Ds_AddMessage (4,6, 'DUMP: '+ImageStr);
                             IF (CurDump > 99)
                                THEN Delete (ImageStr, 1, Length(ImageStr)-2);
                             IF (CurDump < 10)
                                THEN ImageStr := '0' + ImageStr;
                             Ds_SaveImage (FileStr,ImageStr,(Sets.Drive+':'+Sets.Dir+'\'));
                             CurDump := CurDump+Jump;
                           END;
                '1'..'9' : BEGIN
                             s := 0;
                             REPEAT
                               s := s*10 + IVal(Choice);
                               Choice := ReadKey;
                             UNTIL NOT (Choice IN ['0'..'9']);
                             Choice := ' ';
                             CurDump := s;
                           END;
                'K', 'k' : BEGIN {set fixed scale, keep according to current scale}
                             VectSc  := (Hpix/ScrnLen) / Ds_VectSc;
                             MaxV    := DumpMap^[CurDump].MaxVal;
                           END;
                #22      : BEGIN                            {Ctrl-V}
                             CASE Sets.DumpSet OF
                               Stress, Vvect : DISPOSE (Dump);
                               ESS, ISO      : DISPOSE (SnapShot);
                             END; {Case}
                             IF (Sets.DumpSet = ISO)
                                THEN Sets.DumpSet := Stress
                                ELSE Sets.DumpSet := Succ (Sets.DumpSet);
                             CASE Sets.DumpSet OF
                               Stress, Vvect : NEW (Dump);
                               ESS, ISO      : NEW (SnapShot);
                             END; {Case}
                           END;
                'P', 'p' : PlotDump (CurDump);
                'J','j'  : {Jump length}
                           BEGIN
                             Ds_GetInput ('JUMP FACTOR',Sstr); Val (Sstr, Num, Err);
                             IF ((Num >= 1) AND (Num <= QtyDump))
                                THEN Jump := Num;
                           END;
                ELSE CurDump := CurDump+Jump;
              END; {Case}

            Ds_ClearWindow (MsgWind);
            IF (CurDump > QtyDump)
               THEN CurDump := 1;
          END; {While}

        CASE Sets.DumpSet OF
          Stress, Vvect : DISPOSE (Dump);
          ESS, ISO      : DISPOSE (SnapShot);
        END; {Case}
      UNTIL (Ch IN ['Q', 'q', #27]);
    END; {ShowDumps}



  PROCEDURE ShowSeism;
    VAR
      i      : Word;
      Choice : Char;
      Num1, Num2 : Char;
    BEGIN
      Help_context := HelpHist;
      FOR i := 1 TO MaxSeism DO
        NEW (Seism[i].Hist);
      REPEAT
        SetIscreen;
        HistSummary;
        Ds_ResetPalette;
        SetGscreen;
        Ds_GetSeisms;
        Ds_InitSz;
        Ds_GraphSeism;

        REPEAT
          IF (NOT(KeyPressed))
             THEN BEGIN
                    Ds_AddMessage (1,5,'<Z/z>oom  <N>ormal');
                    Ds_AddMessage (2,5,'<X/x>pand <F>ixed');
                    Ds_AddMessage (3,5,'Left/Right/Home/End');
                    Ds_AddMessage (4,5,'<S>ave <P>lot');
                    Ds_AddMessage (5,5,'<M>ore <Q>uit');
                  END;
          Ds_WaitResponse (Choice);
          CASE Choice OF
            'P', 'p' : Pl_PlotSeism;
            'T', 't' : Pl_WriteText;
            'S', 's' : BEGIN {Save}
                         REPEAT
                           Ds_AddMessage (1,7, 'To Save Image, Enter two');
                           Ds_AddMessage (2,7, '  digits (m,n). File will ');
                           Ds_AddMessage (3,7, '  be given extension .Imn.');
                           Ds_AddMessage (4,7, '  (Non-digits are ignored)');
                           Num1 := ReadKey;
                           Num2 := Readkey;
                         UNTIL  ( (Num1 IN ['0'..'9']) AND (Num2 IN ['0'..'9']) );
                         ImageStr := Num1+Num2;
                         FileStr  := Sets.Drive+':'+Sets.Dir+'\'+Sets.Fname;
                         Ds_ClearWindow (MsgWind);
                         Ds_AddMessage (3,6, 'FILE: '+ FileStr);
                         Ds_AddMessage (4,6, 'IMAGE: '+ImageStr);
                         Ds_SaveImage (FileStr,ImageStr,(Sets.Drive+':'+Sets.Dir+'\'));
                         Ds_ClearWindow (MsgWind);
                         Ds_AddMessage (6,7, 'Saved as fn.I'+Num1+Num2);
                       END; {Save}
            'Q', 'q' : ;
            'M', 'm' : ;
            #27      : ;
{            'K', 'k' : DI_CvertPSS (QtySeism);}
            'a'      : DI_CvertAscii (QtySeism,0);
            'A'      : DI_CvertAscii (QtySeism,1);
            ELSE BEGIN  { z, x, i, f, n, #0, etc}
                   Ds_ViewSeism (Choice);
                   Ds_ClearWindow (GraphWind);
                   Ds_GraphSeism;
                 END;
          END; {Case}
        UNTIL (Choice IN ['M', 'm', 'Q', 'q', #27]);

      UNTIL (Choice IN ['Q', 'q', #27]);
      FOR i := 1 TO MaxSeism DO
        DISPOSE (Seism[i].Hist);
    END; {ShowSeism}


  PROCEDURE GraphSingleGeom;
    VAR
      ic         : Word;
      G          : Word;
      Gnum, gID  : LongInt;
      ch         : char;
    BEGIN
      G := CurGeom;
      DI_ReadGeom  (G, Gnum, gID);
      Ds_GeomScale (G, AxDir);
      Ds_GraphGeom (G);
      IF NOT({Sets.Set3D OR }Keypressed) THEN
         CASE TrajOn OF
           1 : Draw_Traject(10000,True,False);
           2 : Draw_Traject(1,False,False);
         END; {if,Case}
      DI_ClearGeom (G);
      ch := '~';
      IF ((QtyCrackD>0) AND (CrackT<>0))
         THEN FOR ic:=1 TO QtyCrackD DO
                IF (ch IN ['~']) THEN
                   BEGIN Ds_GraphCrackD(ic,Slice2D,CrackT,ch); Delay(100); END;
    END; {GraphSingleGeom}


  PROCEDURE GraphMultiGeom;
    VAR
      ic         : Word;
      G           : Word;
      Gnum, gID   : LongInt;
      Fnamebk     : FileName;
      SnapWindBk  : Ds_Window;
      CurSnapBk   : Word;
      Hz, Vt      : Word;
      whstp, wvstp, whsz, wvsz, whst, wvst : Word;
      Vsz, Voff, Hsz, Hoff                 : Integer;
      First, NoLoop : Boolean;
      ch          : char;
    BEGIN
      Traj.NotRead:=True;
      Vsz  := (Vpix-3); Hsz := TRUNC (Vsz/Aspect); Voff := Vsz DIV 15; Hoff := Hsz DIV 25;
      WITH Sets DO
        BEGIN
          whstp := (Hsz-Hoff) DIV whqty;      whsz := TRUNC(0.9*whstp); whst := VtTWind.Fin[1]+1+TRUNC(0.05*whstp);
          wvstp := (Vsz-Voff-Voff) DIV wvqty; wvsz := TRUNC(0.9*wvstp); wvst := HdTWind.Fin[2]+1+TRUNC(0.05*wvstp);
        END;
      FnameBk := Copy (Sets.Fname,1,8);
      SnapWindBk := SnapWindMx;
      CurSnapBk  := CurGeom;

      NoLoop := (TrajOn<>1);
      First  := True;
      ch :='~';
      REPEAT
        WITH Sets DO
          FOR Vt := 0 TO (wvqty-1) DO
           FOR Hz := 0 TO (whqty-1) DO
            BEGIN
              WITH SnapWindMx DO
                BEGIN
                  Sz[1] := whsz;  St[1] := whst + Hz*whstp;  Fin[1] := St[1]+Sz[1]-1;
                  Sz[2] := wvsz;  St[2] := wvst + Vt*wvstp;  Fin[2] := St[2]+Sz[2]-1;
                END; {With SnapWindMx}
              IF (MultFile)
                 THEN BEGIN  Fname := Mfiles[Hz,Vt]; DI_Close; DI_AssignFile (Fname); DI_ReadMap;  END;

              G := CurGeom;
              DI_ReadGeom  (G, Gnum, gID);
              Ds_GeomScale (G, AxDir);
              IF (First) THEN
                 BEGIN  Ds_GraphGeom(G); END;
              IF NOT({Sets.Set3D OR }Keypressed) THEN
                 CASE TrajOn OF
                   1 : IF (First)
                          THEN Draw_Traject(1,True,False)
                          ELSE Draw_Traject(2,True,True);
                   2 : Draw_Traject(1,False,False);
                 END; {if}
              DI_ClearGeom (G);
              DI_ClearTraject;
              IF ((QtyCrackD>0) AND (CrackT<>0))
                 THEN FOR ic:=1 TO QtyCrackD DO
                        IF (ch IN ['~']) THEN
                           BEGIN Ds_GraphCrackD(ic,Slice2D,CrackT,ch); Delay(100); END;
              IF NOT(MultFile)
                 THEN BEGIN  CurGeom := CurGeom + Mjump; IF (CurGeom > QtyGeom) THEN CurGeom := 1;  END;
            END; {For,For,with}
        First := False;
      UNTIL (Keypressed OR NoLoop);

      DI_ClearTraject;
      SnapWindMx := SnapWindBk;
      CurGeom    := CurSnapBk;
      Sets.Fname := Fnamebk;
      IF (Sets.MultFile)
         THEN BEGIN  DI_Close; DI_AssignFile (Sets.Fname); DI_ReadMap;  END;
    END; {GraphMultiGeom}


  PROCEDURE ShowGeoms;
    VAR
      Choice   : Char;
      s        : Word;
    BEGIN
      Help_context := HelpGeom;
      Traj.NotRead:=True;  Traj.TrType:=0;  Traj.Dscale:=False;
      Sstep := False;
      CurGeom := 1;
      Vzoom   := 1;
      Sets.VectSet := False;
      Scale2D.Magn  := 0;
      Sets.Set3D := ( Sets.Set3D AND (GeomMap^ [CurGeom].Model3D) );
      Ang_x := 30;  Ang_y := 30;
      TrajOn := 0;  Tdelay := 10;
      CrackT := 0;                 {Crack dumps initially off}
      AxDir := 0;
      Ds_SetPalette;
      REPEAT
{  CurGeom := 1;}
        Ds_ClearWindow (GraphWind);
        Ds_ClearWindow (LegWind);
        IF (Sets.MultSnap)
           THEN GraphMultiGeom
           ELSE GraphSingleGeom;
        IF (NOT(KeyPressed))
           THEN BEGIN
                  Ds_ClearWindow (MsgWind);
                  Ds_AddMessage (1,6, '<S>ave  <P>lot' );
                  Ds_AddMessage (2,6, '<T>rajects');
                  Ds_AddMessage (3,6, '<V>iew  <D>2d/3d');
                  Ds_AddMessage (4,6, '<Q>uit');
                  Ds_AddMessage (5,6, '(0..9) graph num.');
                  Ds_AddMessage (6,6, '  ... or next page');
                END;
        Ds_WaitResponse (Choice);
        WITH Sets DO
          CASE Choice OF
            #12, #19,'Z','z','L','l','X','x',
            '+','-',
            #0
                     : Common2DChoice(Choice);
            't'      : TrajOn := (TrajOn+1) MOD 3;
            'T'      : BEGIN
                         Traj.TrType  := (Traj.TrType+1) MOD 3;
                         DI_ClearTraject;
                       END;
            'G','g'  : CrackT := (CrackT+1) MOD 4; {Geometry of cracks}
            'R','r'  : BEGIN
                         Traj.Dscale := NOT(Traj.Dscale);
                         DI_ClearTraject;
                       END;
            'c'      : Ds_ResetPalette;
            'C'      : Ds_SetPalette;
            'K', 'k' : BEGIN {set fixed trajectory scale, keep according to current scale}
                         VectSc  := Ds_VectSc;
                       END;
            'S', 's' : BEGIN {Save}
                         Ds_ClearWindow (MsgWind);
                         str (CurGeom,ImageStr);
                         FileStr  := Sets.Drive+':'+Sets.Dir+'\'+Sets.Fname;
                         Ds_AddMessage (3,6, 'FILE: '+ FileStr);
                         Ds_AddMessage (4,6, 'GEOM: '+ImageStr);
                         IF (CurGeom > 99)
                            THEN Delete (ImageStr, 1, Length(ImageStr)-2);
                         IF (CurGeom < 10)
                            THEN ImageStr := '0' + ImageStr;
                         Ds_SaveImage (FileStr,ImageStr,(Sets.Drive+':'+Sets.Dir+'\'));
                         CurGeom := CurGeom+1;
                       END;
            'A', 'a' : BEGIN {2D Slice}
                         Ds_ClearWindow (MsgWind);
                         Ds_AddMessage (3,6, '2D Slice: ');
                         SliceG := True;
                         s := 0;
                         Choice := ReadKey;
                         WHILE (Choice IN ['0'..'9']) DO
                          BEGIN
                            s := s*10 + IVal(Choice);
                            Choice := ReadKey;
                          END;
                         Choice := ' ';
                         Slice2D:=s;
                         IF (s=0) THEN SliceG:=False;
                       END;
            '1'..'9' : BEGIN
                         s := 0;
                         REPEAT
                           s := s*10 + IVal(Choice);
                           Choice := ReadKey;
                         UNTIL NOT (Choice IN ['0'..'9']);
                         Choice := ' ';
                         CurGeom := s;
                       END;
            'D','d'  : IF (GeomMap^ [CurGeom].Model3D)
                          THEN Set3D := NOT(Set3D);
            'V','v'  : IF (GeomMap^ [CurGeom].Model3D)
                          THEN AxDir := (AxDir+1) MOD 3;
{            ' '      : Sstep := NOT(Sstep);}
            ELSE Inc (CurGeom);
          END; {Case}

        IF (CurGeom > QtyGeom)
           THEN CurGeom := 1;
      UNTIL (Choice IN ['Q', 'q', #27]);
      DI_ClearTraject;
    END; {ShowGeoms}


  PROCEDURE RestoreImage;
    VAR
      s          : Word;
      Restored   : Boolean;
      Sstr        : String;
      Num, Err, Jump : Integer;
      First, MenuON  : Boolean;
      VidModebk  : VidModeTypes;
    BEGIN
      SetIscreen;
      ImageNum := 1; Jump := 1; First := True; MenuOn := True;
      DI_RqstImageFile (ImageNum);
      SetGscreen;
      Ds_SetPalette;
      VidModebk := Sets.VidMode;
      REPEAT
{        ImageStr := Strng (ImageNum,0);}
        str (ImageNum,ImageStr);
        IF (ImageNum < 10) THEN ImageStr:='0'+ImageStr;
        FileStr  := Sets.Drive+':'+Sets.Dir+'\'+ImageFname+'.I'+ImageStr;
        Restored := Ds_RestoreImage (FileStr);
        IF (First OR Restored)
           THEN BEGIN
                  First := False;
                  SetColor (DsCol_Fgrnd2);
                  IF (MenuOn)
                     THEN BEGIN
                            Ds_DrawWindow (MsgWind);
                            Ds_DrawWindow (LegWind);
                            Ds_ClearWindow (MsgWind);
                            Ds_AddMessage (2,6, '<c,C>olour Palette');
                            Ds_AddMessage (3,6, '<Q>uit');
                            Ds_AddMessage (4,6, '0..9, New Image');
                            Ds_AddMessage (5,6, 'Else Next Image');
                          END;
                  Ds_AddMessage (1,6, '   IMAGE = '+ImageStr+'    ');
                  Ds_AddMessage (6,6, '  <M>enu On/Off   ');
                END;
        Choice := Ds_ReadKey;
        CASE Choice OF
          'M','m'  : MenuOn := NOT(MenuOn);
          'c'      : Ds_ResetPalette;
          'C'      : Ds_SetPalette;
          'V', 'v' : BEGIN
                       Sets.col256 := NOT (Sets.col256);
                       IF (sets.col256 AND (Sets.VidMode IN [std16]))
                          THEN BEGIN
                                 SetIscreen;
                                 Set_VidMode;
                                 CloseGraph;
                                 SetupGraph;
                                 SetGscreen;
                               END;
                       Set_colours;
                       Ds_SetPalette;
                     END;
          '1'..'9' : BEGIN
                       s := 0;
                       REPEAT
                         s := s*10 + IVal(Choice);
                         Choice := ReadKey;
                       UNTIL NOT (Choice IN ['0'..'9']);
                       ImageNum := s;
                       Choice := '1';
                     END;
          'J','j'  : {Jump length}
                     BEGIN
                       Ds_GetInput ('JUMP FACTOR',Sstr); Val (Sstr, Num, Err);
                       IF ((Num >= 1) AND (Num <= 100))
                          THEN Jump := Num;
                     END;
          ELSE ImageNum := ImageNum+Jump;
        END; {Case}
      UNTIL (Choice IN ['Q','q', #27]);
      Choice := ' ';
      Ds_ClearWindow (MsgWind);
      IF (Sets.VidMode <> VidModebk)
         THEN BEGIN
                Sets.VidMode := VidModebk;
                CloseGraph;
                SetupGraph;
                SetGscreen;
                Ds_ResetPalette;
              END;
    END; {RestoreImage}


  PROCEDURE ShowSlides;
    VAR
      s        : Word;
      Restored : Boolean;
      SingleStep : Boolean;
      Sstr     : String;
      Num, Err, Jump : Integer;
      VidModebk  : VidModeTypes;
    BEGIN
      SingleStep := True;
      SetIscreen;
      ImageNum := 1; Jump := 1;
      DI_RqstImageFile (ImageNum);
      SetGscreen;
      Ds_SetPalette;
      VidModebk := Sets.VidMode;
      SetColor (DsCol_Fgrnd2);
      Ds_DrawWindow (MsgWind);
      Ds_DrawWindow (LegWind);
      Ds_ClearWindow (MsgWind);
      Ds_AddMessage (1,5, '<V>ga Mode');
      Ds_AddMessage (2,5, '<C>olour Palette');
      Ds_AddMessage (3,5, '<Q>uit');
      Ds_AddMessage (4,5, ' 0..9, New Image');
      Ds_AddMessage (5,5, '  Else Next Image');

      Tdelay := 200;
      Choice := 'z'; {Any character which has no effect}
      REPEAT
{        ImageStr := Strng (ImageNum,0);}
        str (ImageNum,ImageStr);
        IF (ImageNum < 10) THEN ImageStr:='0'+ImageStr;
        FileStr  := Sets.Drive+':'+Sets.Dir+'\'+ImageFname+'.I'+ImageStr;
        Restored := Ds_RestoreImage (FileStr);
{        IF (Restored)
           THEN BEGIN
                  SetColor (DsCol_Fgrnd2);
                  Ds_DrawWindow (MsgWind);
                  Ds_DrawWindow (LegWind);
                  Ds_ClearWindow (MsgWind);
                  Ds_AddMessage (2,5, '<C>olour Palette');
                  Ds_AddMessage (3,5, '<Q>uit');
                  Ds_AddMessage (4,5, ' 0..9, New Image');
                  Ds_AddMessage (5,5, '  Else Next Image');
                END;}
{        Ds_AddMessage (1,5, '   IMAGE = '+ImageStr);}

        IF NOT(KeyPressed)
           THEN BEGIN
{                  Choice := 'z'; {Any character which has no effect}
                  IF (Tdelay > 2)
                     THEN IF (Restored)
                             THEN Delay (Tdelay);
                  IF NOT(SingleStep) THEN ImageNum := ImageNum+Jump;
                  IF (ImageNum>90) THEN ImageNum := 1;
                END
           ELSE BEGIN
                  Choice := Ds_ReadKey;
                  CASE Choice OF
                    ' '      : SingleStep := NOT(SingleStep);
                    '+'      : Tdelay := TRUNC (Tdelay * 0.8);         {speed up}
                    '-'      : Tdelay := TRUNC (Tdelay * 1.25 + 1);    {slow down}
                    'c'      : Ds_ResetPalette;
                    'C'      : Ds_SetPalette;
                    'V', 'v' : BEGIN
                                 Sets.col256 := NOT (Sets.col256);
                                 IF (sets.col256 AND (Sets.VidMode IN [std16]))
                                    THEN BEGIN
                                           SetIscreen;
                                           Set_VidMode;
                                           CloseGraph;
                                           SetupGraph;
                                           SetGscreen;
                                         END;
                                 Set_colours;
                                 Ds_SetPalette;
                               END;
                    '1'..'9' : BEGIN
                                 s := 0;
                                 REPEAT
                                   s := s*10 + IVal(Choice);
                                   Choice := ReadKey;
                                 UNTIL NOT (Choice IN ['0'..'9']);
                                 ImageNum := s;
                                 Choice   := '1';
                               END;
                    'J','j'  : {Jump length}
                               BEGIN
                                 Ds_GetInput ('JUMP FACTOR',Sstr); Val (Sstr, Num, Err);
                                 IF ((Num >= 1) AND (Num <= 100))
                                    THEN Jump := Num;
                               END;
                  END; {Case}
                  IF NOT (Choice IN ['C', 'c', 'V', 'v', ' ', '0'..'9'])
                     THEN BEGIN
                            IF (NOT(SingleStep) OR (Choice IN [#13]))
                               THEN ImageNum := ImageNum+Jump;
                            IF (ImageNum>90) THEN ImageNum := 1;
                          END;
                END; {Else}

      UNTIL (Choice IN ['Q','q', #27]);

      Choice := ' ';
      Ds_ClearWindow (MsgWind);
      IF (Sets.VidMode <> VidModebk)
         THEN BEGIN
                Sets.VidMode := VidModebk;
                CloseGraph;
                SetupGraph;
                SetGscreen;
                Ds_ResetPalette;
              END;
    END; {ShowSlides}


  BEGIN {WavePlot}
    ASSIGN  (TmpDebug, 'c:\wavdebug');
    REWRITE (TmpDebug);
    wr_open;

{    REPEAT
      Readln (aa,bb,cc,dd);
      CubeRoots (aa,bb,cc,dd,  r1,r2,r3, r1i,r2i,r3i);
      writeln ('roots: ',r1:12, r2:12, r3:12);
      writeln (' .... Q/q to Quit');
      Choice := Readkey;
    UNTIL (Choice IN ['Q', 'q']);}

    DI_InitVar;
    Tdelay := 0; {1000 mSecs}
    Ds_LinkDrivers;
    Set_InitSets;
    Pl_Init;
    SetupGraph; {Ds_TestWindows; SetupGraph;}
    Ds_SetPalette;
    SetGscreen;
    Help_context := HelpMain;
    MatOn := 1;
{    TestPalette;
    TestPalette2;}
    REPEAT
      Ds_AddMessage (1,4,'<D>isplay Graphs');
      Ds_AddMessage (2,4,'<R>etrieve Images');
      Ds_AddMessage (3,4,'<S>ettings');
      Ds_AddMessage (4,4,'<Q>uit');
      Ds_WaitResponse (Choice);
      CASE Choice OF
        #0       : BEGIN Ch1 := Ds_ReadKey; IF (Ch1=#59) THEN Ds_ShowHelp(Help_context); END;
        'D', 'd' : BEGIN
                     SetIscreen;
                     Help_context := HelpMain;
{                     wr_test;}
                     IF NOT (DI_FileInit)
                        THEN SetGscreen
                        ELSE BEGIN
                               DI_ReadMap;
                               SetGscreen;
                               REPEAT
                                 Ds_AddMessage (1,4,'<S>napShots');
                                 Ds_AddMessage (2,4,'<D>umps');
                                 Ds_AddMessage (3,4,'<H>istories');
                                 Ds_AddMessage (4,4,'<G>eometries');
                                 Ds_WaitResponse (Choice);
                                 CASE Choice OF
                                   'S', 's' : IF (QtySnap > 0) THEN ShowSnaps;
                                   'D', 'd' : IF (QtyDump > 0) THEN ShowDumps;
                                   'H', 'h' : IF (QtyHist > 0) THEN ShowSeism;
                                   'G', 'g' : IF (QtyGeom > 0) THEN ShowGeoms;
                                   #0       : BEGIN Ch1 := Ds_ReadKey; IF (Ch1=#59) THEN Ds_ShowHelp(Help_context); END;
                                 END; {Case}
                               UNTIL (Choice IN ['Q', 'q', #27]);
                               Choice := ' ';
                               DI_Close;
                             END;
                   END;
        'R', 'r' : BEGIN
                     REPEAT
                       Ds_AddMessage (1,2,'<R>estore');
                       Ds_AddMessage (2,2,'<S>lide Show');
                       Ds_WaitResponse (Choice);
                       CASE Choice OF
                         'R', 'r' : RestoreImage;
                         'S', 's' : ShowSlides;
                       END; {Case}
                     UNTIL (Choice IN ['Q', 'q', #27]);
                     Choice := ' ';
                   END;
        'S', 's' : BEGIN
                     SetIscreen;
                     REPEAT
                       Writeln (wr_file,'<Q>uit, snap-<M>ethod, <S>cales, <D>irectory, <V>GA256 <C>olourMode');
                       Writeln (wr_file,'             <F>ourier, d<U>mps <I>mages <R>eset <T>itles (W)indow');
                       wr_flush(wr_nohold);
                       Choice := ReadKey;
                       CASE Choice OF
                         'M', 'm' : BEGIN  Set_ChangeSnap; Set_Method;  END;
                         'S', 's' : Set_Scales;
                         'D', 'd' : DI_RqstPath;
                         'F', 'f' : Set_ChangeFourier;
                         'U', 'u' : Set_ChangeDump;
                         'I', 'i' : Set_ChangeImage;
                         'R', 'r' : Set_Default;
                         'T', 't' : Set_Titles;
                         'W', 'w' : Set_Windows;
                         'V', 'v' : BEGIN
                                      Set_VidMode;
                                      CloseGraph;
                                      SetupGraph;
                                    END;
                         'C', 'c' : Set_ColourMode;
                       END; {Case}
                     UNTIL (Choice IN ['Q', 'q', 'V', 'v', #27]);
                     SetGscreen;
                     Choice := ' ';
                   END;
      END; {Case}
    UNTIL (Choice IN ['Q', 'q']);
    Set_SaveSets (1);
    RestoreCRTMode;
    DI_ClearVar;
    wr_close;
    CLOSE (TmpDebug);
{    ERASE (TmpDebug);}
{        Sound (20); Delay (100); Nosound;}
  END.  {WavePlot}
