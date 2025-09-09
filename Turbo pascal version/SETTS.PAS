{RESOURCE: Settings Resource}

 UNIT Setts;

    {Contains data structures & operators which indicate global settings}

  {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }


  INTERFACE

     TYPE
       ColScaleTypes  = (Col_blackb, Col_whiteb, Col_grey);
       VidModeTypes   = (std16,vga256,svga640,svga800,svga1024,svga16_800,svga16_1024);
       SnapFillTypes  = (Log, Quad, Lin, Fixed, Cosn, DQuad, SqLog);
       LineDirnType   = (Horiz, Vert, Both);
       SnapType       = (ColMap, ContMap, MLine ); {Colour Map, Contour Map, Multi Line}
       DumpType       = (Stress, VVect, ESS, ISO);

     CONST
       Ver           = '407';
       RecLen        =  4096;
       SnapFillNames : ARRAY [SnapFillTypes] OF String[20]
                         = ('Logarithmic', 'Quadratic', 'Linear',  'Fixed', 'Cosine', 'Double-Quad', 'Sqrt-Log');
       MaxLevels     = 256;
{           MaxX          = 50;
            MaxY          = 50;}
       MaxX          = 101;
       MaxY          = 101;
       PenTot        = 8;  {for 7550 plotter}
       MaxSeism      = 8;


     TYPE
       FileName               = String[8];
       Extent                 = RECORD
                                  xp1,yp1, xp2,yp2 : single;
                                END; {Extent}

       SetsType               = RECORD
                                  {Video Mode}
                                  col256       : Boolean;
                                  VidMode      : VidModeTypes;

                                  {File}
                                  Drive        : Char;
                                  Dir          : String;
                                  Fname        : String;

                                  {Fourier}
                                  FreqWind     : Word;   {oversize of freq window as power of 2}
                                  LowFreq      : Word;   {%of max freq for low  cutoff freq}
                                  HighFreq     : Word;   {%of max freq for high cutoff freq}
                                  LowTaper     : Word;   {%of max time-steps for low taper}
                                  HighTaper    : Word;   {%of max time-steps for high taper}
                                  MaxTaper     : Word;   {%of max time-steps for taper}

                                  {Snapshots}
                                  Quick        : Boolean;
                                  Interp       : Boolean;
                                  QuickMax     : Word;
                                  MapOn        : Boolean;
                                  SnapSet      : SnapType;


                                  VGA256dum    : Boolean;  {dummy, not used anymore}


                                  {Multi-Lines}
                                  Dirn         : LineDirnType;
                                  LineCol      : Boolean;
                                  SameColour   : Boolean;

                                  {Dumps}
                                  DumpSet      : DumpType;

                                  {Images}
                                  Compress     : Boolean;

                                  {Histories}
                                  SLegMax      : ARRAY [1..MaxSeism] OF Single;
                                  Title        : ARRAY [1..4] OF String[40];
                                  HzTitle      : String[40];
                                  VtTitle      : String[40];
                                  HdTitle      : String[40];

                                  {ESS}
                                  Thet, Frict  : Single;

                                  {Geom}
                                  Set3D        : Boolean;
                                  ang_3D       : Integer;

                                  {Multi snapshots}
                                  MultSnap     : Boolean;
                                  MultFile     : Boolean;
                                  whqty, wvqty : Word;
                                  Mjump        : Word;
                                  Mfiles       : ARRAY [0..5,0..5] OF FileName;
                                  Msnap        : ARRAY [0..5,0..5] OF Word;

                                  {Scaling}
                                    {Geom Scale}
                                      GeomScSet  : Boolean;
                                      GeomSc     : Single;
                                      GeomExtSet : Boolean;
                                      GeomExt    : Extent;

                                    {V Scale}
                                      VectSet      : Boolean;
                                      VectSc       : Single;
                                      MaxV         : Single;
                                      MinV         : Single;

                                    {Colour Scaling}
                                      SnapFill     : SnapFillTypes;
                                      SnapColAbs   : Boolean;
                                      Snap1Lev     : Single;                  {Ratio over 1 level}
                                      SnapNLev     : Single;                  {Ratio over All levels}

                                    {Colour Maps}
                                      QtyLevels    : Word;
                                      QtyCols      : Word;

                                      ShowTitles   : Boolean;
                                      ColScale     : ColScaleTypes;

                                END;

     VAR
       Sets               : SetsType;
       TmpDebug           : Text;
       QtySets, CurSets   : Word;
       LevListMx          : Word;
       MultNoLink         : Boolean;
       Slice2D            : Word;
       SliceG             : Boolean;
       Sstep              : Boolean;
       Seis_HMax,Seis_HMin  : Single;
       MatOn              : 0..3;

     PROCEDURE Set_VidMode;

     PROCEDURE Set_ColourMode;

     PROCEDURE Set_Colours;

     PROCEDURE Set_ReadSets (RecNum : Word);

     PROCEDURE Set_SaveSets (RecNum : Word);

     PROCEDURE Set_InitSets;

     PROCEDURE Set_Default;

     PROCEDURE Set_Method;

     PROCEDURE Set_Scales;

     PROCEDURE Set_LegMax;

     PROCEDURE Set_ChangeFourier;

     PROCEDURE Set_ChangeSnap;

     PROCEDURE Set_ChangeImage;

     PROCEDURE Set_ChangeDump;

     PROCEDURE Set_Titles;

     PROCEDURE Set_Windows;


{ * * * * * * * * * * * * * * * * * * * * * * * *}
     {Silly bit of code, to allow writeln output in graphics mode}
     TYPE
       wr_holdType = (wr_hold, wr_nohold, wr_nohold2);
     VAR
       wr_file : Text;
       wr_curl : word;
     PROCEDURE wr_open;
     PROCEDURE wr_close;
     PROCEDURE wr_clear;
     PROCEDURE wr_flush (hold:wr_holdType);
     PROCEDURE wr_test;

{ * * * * * * * * * * * * * * * * * * * * * * * *}


  IMPLEMENTATION

    USES Graph, Crt, Funct;

     PROCEDURE Set_VidMode;
       VAR
         Vmode : Char;
       BEGIN
         wr_clear; Writeln (wr_file);
         Writeln('Which video mode would you like to use?');
         Writeln(wr_file,'  0) Standard 16 colour modes: VGA,EGA,CGA etc.');
         Writeln(wr_file,'  1) Standard VGA - 256 colour, 320x200');
         Writeln(wr_file,'  2) SVGA: 640x480,  256 colour');
         Writeln(wr_file,'  3) SVGA: 800x600,  256 colour');
         Writeln(wr_file,'  4) SVGA: 1024x768, 256 colour');
{         Writeln(wr_file,'  5) SVGA: 800x600,   16 colour');
         Writeln(wr_file,'  6) SVGA: 1024x768,  16 colour');}
         Write(wr_file,'> ');
         wr_flush(wr_nohold); Vmode := ReadKey;
         CASE Vmode OF
           '0' : Sets.VidMode := std16;
           '1' : Sets.VidMode := vga256;
           '2' : Sets.VidMode := svga640;
           '3' : Sets.VidMode := svga800;
           '4' : Sets.VidMode := svga1024;
           '5' : Sets.VidMode := svga16_800;
           '6' : Sets.VidMode := svga16_1024;
           ELSE Sets.VidMode := std16;
         END; {Case}
         IF (Sets.VidMode > vga256)
            THEN BEGIN
                   Writeln(wr_file);
                   Writeln(wr_file, '    * * WARNING !! * *');
                   Writeln(wr_file, '      You have selected a Super VGA mode. If your computer does not');
                   Writeln(wr_file, '      support this mode, it could damage your equipment.');
                   Writeln(wr_file, '         Press (Y/y) to continue, or any other key to cancel');
                   wr_flush(wr_nohold);
                   IF NOT(ReadKey IN ['Y','y'])
                      THEN Sets.VidMode := std16;
                 END;
{         Writeln(wr_file,'  0) 320x200x256');}
{         Writeln(wr_file,'  1) 640x400x256');}
{         Writeln(wr_file,'  5) 640x350x256');}
{         Writeln(wr_file,'  6) 1280x1024x256');}
       END; {Set_VidMode}

     PROCEDURE Set_ColourMode;
       VAR
         Ch : Char;
       BEGIN
         WITH Sets DO
           REPEAT
             Writeln (wr_file,'COLOUR MODE: ');
             Write   (wr_file,'  (H)i_col  (L)o_col  (B)lack b/g  (W)hite_b/g  (G)reyscale  (Q)uit');
             wr_flush(wr_nohold); Ch := ReadKey;
             CASE Ch OF
               'H', 'h' : BEGIN
                            col256 := col256 AND (VidMode <> std16);
                          END;
               'L', 'l' : BEGIN
                            col256 := False;
                          END;
               'B', 'b' : BEGIN
                            ColScale := Col_blackb;
                          END;
               'W', 'w' : BEGIN
                            ColScale := Col_whiteb;
                          END;
               'G', 'g' : BEGIN
                            ColScale := Col_grey;
                          END;
             END; {Case}
           UNTIL ( NOT(Ch IN ['H','h','L','l','B','b','W','w','G','g']) );
       END; {Set_ColourMode}


     PROCEDURE Set_colours;
       BEGIN
         WITH Sets DO
           IF NOT(Sets.col256)
              THEN BEGIN  QtyCols:=13; QtyLevels:=11;
                          SnapNLev:=256; Snap1Lev:=Exp(ln(SnapNLev)/QtyLevels);
                   END
              ELSE CASE VidMOde OF
                      vga256 : BEGIN
                                 QtyCols:=48; QtyLevels:=48;
                                 SnapNLev:=256; Snap1Lev:=Exp(ln(SnapNLev)/QtyLevels);
                               END;
                      svga640,svga800,svga1024 : BEGIN
                                 QtyCols:=71; QtyLevels:=71;
                                 SnapNLev:=256; Snap1Lev:=Exp(ln(SnapNLev)/QtyLevels);
                               END;
                          ELSE
                               BEGIN
                                 QtyCols:=13; QtyLevels:=11;
                                 SnapNLev:=256; Snap1Lev:=Exp(ln(SnapNLev)/QtyLevels);
                               END;
                   END; {Case}
       END;  {Set_colours}


     PROCEDURE Set_Default;
       VAR
         i      : Integer;
         Hz, Vt : Word;
       BEGIN
         {Variables not permanently recorded}
          MultNoLink   := False;
          SliceG       := True;
          Slice2D      := 0;
          Sstep        := False;
          Seis_HMin    := 0;
          Seis_HMax    := 1;
         WITH Sets DO
           BEGIN
             LevListMx    := 11;
             Drive        := 'C';
             Dir          := '.';
             Fname        := 'WAVE';
             FreqWind     := 1;
             LowFreq      := 10;
             HighFreq     := 100;
             LowTaper     := 5;
             HighTaper    := 95;
             MaxTaper     := 5;


             SnapFill     := Lin;
             SnapColAbs   := False;
             Snap1Lev     := 2;
             SnapNLev     := 256;

             Col256       := False;
             Vidmode      := std16;            {256col, 640x480}
             ColScale     := Col_blackb;
             QtyLevels    := 11;
             QtyCols      := 11;
             Quick        := False;
             Interp       := False;
             QuickMax     := MaxX;
             MapOn        := True;
             SnapSet      := ColMap;
             Dirn         := Horiz;
             LineCol      := True;
             SameColour   := False;
             DumpSet      := Stress;
             VectSc       := 1e-6;
             VectSet      := False;
             MaxV         :=  1e6;
             MinV         := -1e6;
             Compress     := True;
             FOR i := 1 TO MaxSeism DO
               SLegMax[i] := 1e6;
             HzTitle := '';
             VtTitle := '';
             HdTitle := '';
             FOR i := 1 TO 4 DO
               Title[i] := '';

             Thet    := 0; Frict   := 0;

             ang_3D := 30;
             GeomScSet  := False; GeomSc := 0;
             GeomExtSet := False;
             WITH GeomExt DO
               BEGIN xp1:=0; yp1:=0; xp2:=100; yp2:=100; END;

             MultSnap := False; MultFile := False; whqty := 1; wvqty := 1; Mjump := 1;
             ShowTitles := True;
             FOR Vt := 0 TO 5 DO
              FOR Hz := 0 TO 5 DO
               BEGIN
                 Msnap [Hz,Vt] := 1;
                 Mfiles[Hz,Vt] := '';
               END;
           END;
       END; {Set_Default}

     PROCEDURE Set_ReadSets (RecNum : Word);
       VAR
         SetsFile : File;
         Sz       : Word;
       BEGIN
         IF (RecNum <= QtySets)
            THEN BEGIN
                   ASSIGN (SetsFile, 'Wvpl'+Ver+'.Set');
                   Sz := SizeOf (Sets);
                   {$I-}
                     RESET (SetsFile,1);
                     Seek (SetsFile, RecLen * (RecNum - 1) );
                     BlockRead (SetsFile, Sets, Sz);
                     CLOSE (SetsFile);
                     WHILE (IOResult <> 0) DO;
                     CurSets := RecNum;
                   {$I+}
                 END;
       END; {Set_ReadSets}

     PROCEDURE Set_SaveSets (RecNum : Word);
       VAR
         SetsFile : File;
         Sz       : Word;
       BEGIN
         IF (RecNum > QtySets)
            THEN BEGIN
                   Inc (QtySets);
                   RecNum := QtySets;
                 END;
         ASSIGN (SetsFile, 'Wvpl'+Ver+'.Set');
         Sz := SizeOf (Sets);
         {$I-}
           RESET (SetsFile, RecLen);
           Seek (SetsFile, (RecNum - 1) );
           BlockWrite (SetsFile, Sets, 1);
           CLOSE (SetsFile);
           WHILE (IOResult <> 0) DO;
           CurSets := RecNum;
         {$I+}
       END; {Set_SaveSets}

     PROCEDURE Set_InitSets;
       VAR
         SetsFile : File;
         Sz       : LongInt;
       BEGIN
         Set_Default;
         ASSIGN (SetsFile, 'Wvpl'+Ver+'.Set');
         Sz := SizeOf (Sets);
         {$I-}
           RESET (SetsFile, 1);
           QtySets := FileSize (SetsFile) DIV RecLen;
           CLOSE (SetsFile);
           WHILE (IOResult <> 0) DO;
         {$I+}
         IF (QtySets >= 1)
            THEN Set_ReadSets (1)
            ELSE BEGIN
                   QtySets := 1; CurSets := 1;
                   Writeln ('Errors were encountered reading the saved settings file (WVPL'+Ver+'.SET) !');
                   Writeln ('       The Default Settings will be used');
                   Writeln ('       Use Menu later to Alter & Save New Settings');
                   Writeln; Writeln (wr_file,' -- PRESS ENTER TO CONTINUE --- ');
                   Readln;
                   Set_Default;
                   {$I-}
                     REWRITE (SetsFile, RecLen);
                     BlockWrite (SetsFile, Sets, 1);
                     CLOSE (SetsFile);
                     WHILE (IOResult <> 0) DO;
                   {$I+}
                 END;
       END; {Set_InitSets}

     PROCEDURE Set_Method;
       BEGIN
         Writeln(wr_file);
         Write  (wr_file,'  (I)nterpolate Snaps (Y/N):      '); wr_flush(wr_nohold);
         IF (ReadKey IN ['Y', 'y'])
            THEN Sets.Interp := True
            ELSE Sets.Interp := False;
         IF (Sets.Interp)
            THEN outtext ('[INTERPOLATE]  ')
            ELSE outtext ('[DON"T INTERPOLATE]  ');
         Writeln(wr_file);
         Write  (wr_file,'  (Q)uick or (C)omplete Snaps :   '); wr_flush(wr_nohold);
         IF (ReadKey IN ['C', 'c'])
            THEN Sets.Quick := False
            ELSE Sets.Quick := True;
         IF (Sets.Quick)
            THEN outtext ('[QUICK]  ')
            ELSE outtext ('[COMPLETE]  ');
         IF (Sets.Quick)
            THEN BEGIN
                   Write (wr_file,'  (M)ap or average full snapshot (Y/N) :   '); wr_flush(wr_nohold);
                   IF (ReadKey IN ['Y', 'y'])
                      THEN Sets.MapOn := True
                      ELSE Sets.MapOn := False;
                   IF (Sets.MapOn)
                      THEN outtext ('[MAP/AVERAGE]    ')
                      ELSE outtext ('[DON"T AVERAGE]  ');
{                   Writeln(wr_file);}
                 END;
         IF (Sets.Quick OR Sets.Interp)
            THEN BEGIN
                   Write (wr_file, 'Max elements to use for Quick retrieval (', Strng(Sets.QuickMax,0) + ') : ');
                   wr_flush(wr_nohold);
                   ReadLn (Sets.QuickMax);
                   IF (Sets.QuickMax > MaxX)
                      THEN Sets.QuickMax := MaxX;
                   IF (Sets.QuickMax < 3)
                      THEN Sets.QuickMax := 3;
                 END;
         Writeln(wr_file);
       END; {Set_Method}


     PROCEDURE Set_LegMax;
       VAR
         i   : Integer;
         Dum : Single;
       BEGIN
         i := 1;
         WITH Sets DO
           REPEAT
             Writeln (wr_file,'Enter Maximum Vertical Axis legends for Seismograms');
             Writeln (wr_file);
             Write (wr_file,'Enter LegMax[',i:1,'] ',SLegMax[i]:10, ' - (0) ends. ');
             wr_flush(wr_nohold); ReadLn(Dum);
             IF (Dum <> 0)
                THEN SLegMax[i] := Dum;
             Inc(i);
           UNTIL ( (Dum=0) OR (i > MaxSeism) );
           Writeln (wr_file); wr_flush(wr_nohold);
           Writeln (wr_file,'Enter Minimum Horizontal Axis legends for Seismograms');
           Writeln (wr_file); wr_flush(wr_nohold); ReadLn(Seis_HMin);
           Writeln (wr_file,'Enter Maximum Horizontal Axis legends for Seismograms');
           Writeln (wr_file); wr_flush(wr_nohold); ReadLn(Dum);
           IF (Dum > Seis_HMin)
              THEN Seis_HMax := Dum
              ELSE Seis_HMax := Seis_HMin+1;
       END; {Set_LegMax}

     PROCEDURE Set_ChangeFourier;
       VAR
         i   : Integer;
         Dum : Single;
       BEGIN
         WITH Sets DO
           REPEAT
             Write (wr_file,'Enter Lower Frequency for Filter (x%): ');
             wr_flush(wr_nohold); ReadLn (LowFreq);
             Write (wr_file,'Enter Upper Frequency for Filter (x%): ');
             wr_flush(wr_nohold); ReadLn (HighFreq);
           UNTIL ( (LowFreq >= 0) AND (LowFreq <= 100) AND
                   (HighFreq >= 0) AND (HighFreq <= 100)   );
         WITH Sets DO
           REPEAT
             Write (wr_file,'Enter oversize for time windows for Fourier Transforms: ');
             wr_flush(wr_nohold); ReadLn (FreqWind);
           UNTIL (FreqWind >= 0);
         WITH Sets DO
           REPEAT
             Write (wr_file,'Enter Lower Taper length (x% of time-steps): ');
             wr_flush(wr_nohold); ReadLn (LowTaper);
             Write (wr_file,'Enter Upper Taper length (x% of time-steps): ');
             wr_flush(wr_nohold); ReadLn (HighTaper);
             Write (wr_file,'Enter Maximum Taper length (x% of time-steps): ');
             wr_flush(wr_nohold); ReadLn (MaxTaper);
             IF ((MaxTaper<0) OR (MaxTaper>100)) THEN MaxTaper:=5;
           UNTIL ( (LowTaper >= 0) AND (LowTaper <= 100) AND
                   (HighTaper >= 0) AND (HighTaper <= 100)   );
       END; {Set_ChangeFourier}

     PROCEDURE Set_ChangeSnap;
       BEGIN
         Writeln(wr_file);
         Write (wr_file,'  (M)ap-colours   (C)ontours    (L)ines ');
         CASE Sets.SnapSet OF
            ColMap  : Write (wr_file,'[MAPCOL]:  ');
            MLine   : Write (wr_file,'[LINES]:   ');
            ContMap : Write (wr_file,'[CONT]:    ');
         END; {Case}
         wr_flush(wr_nohold);
         CASE ReadKey OF
           'M', 'm' : Sets.SnapSet := ColMap;
           'L', 'l' : Sets.SnapSet := MLine;
           'C', 'c' : Sets.SnapSet := ContMap;
         END; {Case}
         CASE Sets.SnapSet OF
            ColMap  : outtext ('[MAPCOL]:  ');
            MLine   : outtext ('[LINES]:   ');
            ContMap : outtext ('[CONT]:    ');
         END; {Case}
       END; {Set_ChangeSnap}

     PROCEDURE Set_ChangeImage;
       BEGIN
         Writeln (wr_file);
         Write   (wr_file,'  Compress Images ? (Y/N) '); wr_flush(wr_hold);
         CASE ReadKey OF
           'Y', 'y' : Sets.Compress := True;
           'N', 'n' : Sets.Compress := False;
         END; {Case}
       END; {Set_ChangeImage}

     PROCEDURE Set_ChangeDump;
       BEGIN
         Writeln (wr_file);
         Write (wr_file,'  (S)tress Axes   (V)elocity Vectors  (E)SS  (I)sochrom ');
         CASE Sets.DumpSet OF
            Stress : outtext ('[STRESS]:  ');
            Vvect  : outtext ('[VVECT]:   ');
            ESS    : outtext ('[ESS]: ');
            ISO    : outtext ('[ISOCHROMATICS]: ');
         END; {Case}
         CASE ReadKey OF
           'S', 's' : Sets.DumpSet := Stress;
           'V', 'v' : Sets.DumpSet := Vvect;
           'E', 'e' : Sets.DumpSet := ESS;
           'I', 'i' : Sets.DumpSet := ISO;
         END; {Case}
         IF (Sets.DumpSet = ESS)
            THEN BEGIN
                   writeln (wr_file);
                   Write (wr_file,'Enter Friction Angle for ESS calc  [', Sets.Frict:6, ']:  ');
                   wr_flush(wr_nohold); Readln (Sets.Frict);
                   Write (wr_file,'Enter Rotation Angle for ESS calc  [', Sets.Thet:6 , ']:  ');
                   wr_flush(wr_nohold); Readln (Sets.Thet);
                 END;
         writeln (wr_file);
       END; {Set_ChangeDump}


     PROCEDURE Set_ContourScale;
       VAR
         Ch : Char;

       PROCEDURE Set_SnapFill;
         VAR
           Ch : Char;
         BEGIN
           WITH Sets DO
             BEGIN
               Writeln (wr_file);
               Writeln (wr_file,'Set the type of fill for a snapshot (contour) plot');
               writeln (wr_file,'    (This refers to the colour distribution)');
               Writeln (wr_file,'    (l)og,    (Q)uadratic,   (L)inear,        (F)ixed levels, ');
               Write   (wr_file,'    (C)osine, (D)ouble Quad, (S)quareRoot-log   [ ' + SnapFillNames[SnapFill] + ']');
               wr_flush(wr_nohold);
               CASE ReadKey OF
                 'l'      : SnapFill := log;
                 'Q', 'q' : SnapFill := Quad;
                 'L'      : SnapFill := Lin;
                 'F', 'f' : SnapFill := Fixed;
                 'C', 'c' : SnapFill := Cosn;
                 'D', 'd' : SnapFill := DQuad;
                 'S', 's' : SnapFill := SqLog;
               END; {Case}
               Writeln (wr_file,SnapFillNames[SnapFill]:40); wr_flush(wr_nohold);
               IF (SnapFill = log)
                  THEN BEGIN
                         Write (wr_file,'Fix Level (Q)uanta or (R)ange - for Log distributions (Q/R)? ');
                         wr_flush(wr_nohold); Ch := ReadKey;
                         CASE Ch OF
                           'Q', 'q' :
                              BEGIN
                                Write (wr_file,'Enter Quanta/ Ratio between levels (' + Strng(Snap1Lev,$0300) + ') : ');
                                wr_flush(wr_nohold); ReadLn (Snap1Lev);
                                SnapNLev := 0;
                              END;
                           'R', 'r' :
                              BEGIN
                                Write (wr_file,'Enter Range of variation of all levels (' + Strng(SnapNLev,$0300) + ') : ');
                                wr_flush(wr_nohold); ReadLn (SnapNLev);
                                Snap1Lev := Exp (Ln(SnapNLev) / QtyLevels);
                              END;
                         END; {Case}
                       END;
             END; {With}
           writeln(wr_file);
         END; {Set_SnapFill}

       BEGIN {Set_ContourScale}
         WITH Sets DO
           REPEAT
             Writeln (wr_file,'CONTOUR SCALES: ');
             Write   (wr_file,'  (F)ill-Type  (L)evels  (C)olours  fi(X)  (U)nfix  (Q)uit');
             wr_flush(wr_nohold); Ch := ReadKey; {outtext(Ch);}
             CASE Ch OF
               'F', 'f' : BEGIN
                            Set_SnapFill;
                          END;
               'L', 'l' : BEGIN
                            Write (wr_file,'Enter Qty of Levels  (' + Strng(QtyLevels,0) + ') : ');
                            wr_flush(wr_nohold); ReadLn (QtyLevels);
                            Write (wr_file,'Enter Qty of Levels to display (' + Strng(LevListMx,0) + ') : ');
                            wr_flush(wr_nohold); ReadLn (LevListMx);
                          END;
               'C', 'c' : BEGIN
                            Write (wr_file,'Enter Qty of Colours (' + Strng(QtyCols,0) + ') : ');
                            wr_flush(wr_nohold); ReadLn (QtyCols);
                          END;
               'X', 'x' : BEGIN
                            VectSet := True;
                            Write (wr_file,'Enter Max for Colour Scale [', MaxV:6, ']:  ');
                            wr_flush(wr_nohold); Readln (MaxV);
                            Write (wr_file,'Enter Min for Colour Scale [', MinV:6, ']:  ');
                            wr_flush(wr_nohold); Readln (MinV);
                          END;
               'U', 'u' : BEGIN
                            VectSet := False
                          END;
             END; {Case}
{             writeln;}
           UNTIL ( NOT(Ch IN ['F','f','L','l','C','c','X','x','U','u']) );
       END; {Set_ContourScale}


     PROCEDURE Set_Scales;
       VAR
         Ch : Char;
       BEGIN
         WITH Sets DO
           REPEAT
             Writeln (wr_file,' CHANGE SCALES: ');
             Write   (wr_file,'  (G)eometric  (V)ector  (C)ontour  (H)istories  (Q)uit');
             wr_flush(wr_nohold); Ch := ReadKey; {outtext(Ch);} writeln(wr_file);
             CASE Ch OF
               'G', 'g' : BEGIN
                            Write (wr_file,' Fix Geometric Scale (Y/N) ?'); wr_flush(wr_nohold);
                            IF NOT (ReadKey IN ['Y','y'])
                               THEN BEGIN  GeomScSet := False; END
                               ELSE BEGIN
                                      GeomScSet := True;
                                      Write (wr_file,'Enter Plotter Scale - 1CM represents [', GeomSc:6, ']:  ');
                                      wr_flush(wr_nohold); Readln (GeomSc);
                                    END; {else}

                            Write (wr_file,' Fix Geometric Extents (Y/N) ?'); wr_flush(wr_nohold);
                            IF NOT (ReadKey IN ['Y','y'])
                               THEN GeomExtSet := False
                               ELSE BEGIN
                                      GeomExtSet := True;
                                      WITH GeomExt DO
                                        BEGIN
                                          Write (wr_file,'Enter Graph Extents:  [Xp1 = ', xp1:8:2,']  ');
                                          wr_flush(wr_nohold); Readln (xp1);
                                          Write (wr_file,'                   :  [Yp1 = ', yp1:8:2,']  ');
                                          wr_flush(wr_nohold); Readln (yp1);
                                          Write (wr_file,'                   :  [Xp2 = ', xp2:8:2,']  ');
                                          wr_flush(wr_nohold); Readln (xp2);
                                          Write (wr_file,'                   :  [Yp2 = ', yp2:8:2,']  ');
                                          wr_flush(wr_nohold); Readln (yp2);
                                          writeln (wr_file);
                                        END; {with}
                                    END; {else}

                          END; {Case Geom}

               'V', 'v' : BEGIN
                            Write (wr_file,' Fix Vector Scale (Y/N) ?'); wr_flush(wr_nohold);
                            IF NOT (ReadKey IN ['Y','y'])
                               THEN VectSet := False
                               ELSE BEGIN
                                      writeln (wr_file);
                                      VectSet := True;
                                      IF (VectSc = 0) THEN VectSc := 1;
                                      Write (wr_file,'Enter Plotter Scale - 1CM represents [', 1/VectSc:6, ']:  ');
                                      wr_flush(wr_nohold); Readln (VectSc);
                                      IF (VectSc = 0) THEN VectSc := 1;
                                      VectSC := 1 / VectSC;
                                    END;
                          END;
               'C', 'c' : BEGIN
                            Set_ContourScale;
                          END;
               'H', 'h' : BEGIN
                            Set_LegMax;
                          END;
             END; {Case}
             writeln (wr_file);
           UNTIL ( NOT(Ch IN ['G','g','V','v','C','c','H','h']) );
         writeln (wr_file);
       END; {Set_Scales}

     PROCEDURE Set_Titles;
       VAR
         Ch     : Char;
         NTitle : String;
         i      : Integer;
       BEGIN
         WITH Sets DO
           REPEAT
             Writeln (wr_file,'CHANGE TITLES: ');
             Write   (wr_file,'  (M)ain   (H)z-Axis   (V)t-Axis   (I)nfo   (S)how   (O)ff   (Q)uit');
             wr_flush(wr_nohold); Ch := ReadKey; {outtext(Ch);}
             CASE Ch OF
               'S', 's' : BEGIN
                            Writeln (wr_file,'      ** Titles ON **');
                            ShowTitles := True;
                          END;
               'O', 'o' : BEGIN
                            Writeln (wr_file,'      ** Titles OFF **');
                            ShowTitles := False;
                          END;
               'H', 'h' : BEGIN
                            Writeln (wr_file,'Horizontal Title (Old): ', HzTitle);
                            Write   (wr_file,'              (Change): ');
                            wr_flush(wr_nohold); Readln (NTitle);
                            IF (NTitle <> '')
                               THEN HzTitle := Copy (NTitle, 1, 40);
                            Writeln (wr_file,'Horizontal Title (New): ', HzTitle); Writeln(wr_file);
                          END;
               'V', 'v' : BEGIN
                            Writeln (wr_file,'Vertical Title (Old): ', VtTitle);
                            Write   (wr_file,'            (Change): ');
                            wr_flush(wr_nohold); Readln (NTitle);
                            IF (NTitle <> '')
                               THEN VtTitle := Copy (NTitle, 1, 40);
                            Writeln (wr_file,'Vertical Title (New): ', VtTitle); Writeln(wr_file);
                          END;
               'M', 'm' : BEGIN
                            Writeln (wr_file,'Main Heading (Old): ', HdTitle);
                            Write   (wr_file,'             (Change): ');
                            wr_flush(wr_nohold); Readln (NTitle);
                            IF (NTitle <> '')
                               THEN HdTitle := Copy (NTitle, 1, 40);
                            Writeln (wr_file,'Main Heading (Old): ', HdTitle);
                          END;
               'T', 't' : BEGIN
                            Writeln (wr_file,'Title Window:', HzTitle);
                            FOR i := 1 TO 4 DO
                              BEGIN
                                Writeln (wr_file,'   Title ',i:1,' (Old): ', Title[i]);
                                Write   (wr_file,'              (Change): ');
                                wr_flush(wr_nohold); Readln (NTitle);
                                IF (NTitle <> '')
                                   THEN Title[i] := Copy (NTitle, 1, 30);
                                Writeln (wr_file,'   Title ',i:1,' (New): ', Title[i]); Writeln(wr_file);
                              END; {For}
                            Writeln(wr_file);
                          END;
             END; {Case}
             wr_flush(wr_nohold);
           UNTIL ( NOT(Ch IN ['H','h','V','v','T','t','M','m']) );
       END; {Set_Titles}

     PROCEDURE Set_Windows;
       VAR
         i   : Integer;
         Dum : Single;
         Hz, Vt : Word;
         Name   : FileName;
       BEGIN
         i := 1;
         WITH Sets DO
           BEGIN
             Write (wr_file,'Multiple Snapshot Windows (Y/N)');
             wr_flush(wr_nohold);
             IF (ReadKey IN ['Y','y'])
                THEN MultSnap := True
                ELSE MultSnap := False;
             writeln(wr_file);
             IF (MultSnap)
                THEN BEGIN
                       Write (wr_file,'Number of Horizontal Windows (Max 6)  [', whqty:4, ']:  ');
                       wr_flush(wr_nohold); Readln (whqty);
                       Write (wr_file,'Number of Vertical   Windows (Max 6)  [', wvqty:4, ']:  ');
                       wr_flush(wr_nohold); Readln (wvqty);
                       IF (whqty > 6) THEN whqty := 5; IF (whqty < 1) THEN whqty := 1;
                       IF (wvqty > 6) THEN wvqty := 5; IF (wvqty < 1) THEN wvqty := 1;
                       IF ((whqty=1) AND (wvqty=1)) THEN MultSnap := False;
                     END;
             IF (MultSnap)
                THEN BEGIN
                       Write (wr_file,'Multiple  (F)iles or (S)napshots'); wr_flush(wr_nohold);
                       IF NOT(ReadKey IN ['F','f'])
                          THEN BEGIN
                                 writeln(wr_file);
                                 Write  (wr_file,'Increment between snapshots in multiple views  [', Mjump:4, ']:  ');
                                 wr_flush(wr_nohold); Readln (Mjump);
                                 MultFile := False;
                                 IF (Mjump = 0) THEN
                                    FOR Vt := 0 TO (wvqty-1) DO
                                     FOR Hz := 0 TO (whqty-1) DO
                                      BEGIN
                                        Write (wr_file,'Snapshot for window (',Hz:2,Vt:2,')  [', Msnap[Hz,Vt], ']:  ');
                                        wr_flush(wr_nohold); Readln (Msnap[Hz,Vt]);
                                      END;
                               END
                          ELSE BEGIN
                                 writeln(wr_file);
                                 MultFile := True;
                                 FOR Vt := 0 TO (wvqty-1) DO
                                  FOR Hz := 0 TO (whqty-1) DO
                                   BEGIN
                                     Write (wr_file,'File for window (',Hz:2,Vt:2,')  [', Mfiles[Hz,Vt], ']:  ');
                                     wr_flush(wr_nohold); Readln (Name);
                                     IF (Name <> '') THEN Mfiles[Hz,Vt]:=Name;
                                     Write (wr_file,'Snapshot for window (',Hz:2,Vt:2,')  [', Msnap[Hz,Vt], ']:  ');
                                     wr_flush(wr_nohold); Readln (Msnap[Hz,Vt]);
                                   END;
                               END; {Else}
                     END; {If MultSnap}
           END; {With}
       END; {Set_Windows}


{ * * * * * * * * * * * * * * * * * * * * * * * *}
     {Silly bit of code, to allow writeln output in graphics mode}
     PROCEDURE wr_open;
       BEGIN
         ASSIGN (wr_file, 'wr.tmp');
         Rewrite (wr_file);
         wr_curl := 1;
       END; {wr_open}

     PROCEDURE wr_close;
       BEGIN
         close (wr_file);
         erase (wr_file);
       END; {wr_close}

     PROCEDURE wr_clear;
       BEGIN {Set up an info window}
         ClearViewPort;
         gotoxy (1,1);
         SetTextJustify (LeftText, TopText);
         wr_curl := 1;
       END; {wr_clear}

     PROCEDURE wr_flush (hold:wr_holdType);
       CONST
         s1 = '                                                          .. more ..';
       VAR
         s : string;
         wr_theight, wr_height : word;
         Voff : word;
         Ch   : Char;
       BEGIN
         IF (wr_curl = 1) THEN wr_clear;
         wr_theight := ROUND(textheight('ABC')*1.6);
         wr_theight := (GetMaxY+1) DIV 30;
         Voff := ((GetMaxY+1)-480) DIV 60;
         wr_height  := 16;
         Reset (wr_file);
         WHILE (NOT(Eof(wr_file))) DO
           BEGIN
             readln (wr_file, s);
             moveto (10,wr_curl*wr_height-2-Voff);
             outtext (s);
             inc(wr_curl);
             IF (wr_curl >= 30) THEN wr_clear;
           END;
         IF ((hold=wr_nohold) AND (wr_curl >= 22))
            THEN BEGIN  wr_curl:=1; END;
         IF ((hold=wr_nohold2) AND (wr_curl >= 22))
            THEN BEGIN
                   moveto (10,wr_curl*wr_height-2-Voff); outtext (s1);
                   wr_curl:=1; Ch := ReadKey;
                 END;
         gotoxy (((Getx+1) DIV 8)+2, ROUND((Gety+1)*0.0625)+1);
         Rewrite (wr_file);
       END; {wr_flush}

     PROCEDURE wr_test;
       VAR
         i : Integer;
         r : single;
         s : string;
         c : char;
       BEGIN
         writeln (wr_file, '(1) testing (integer): '); wr_flush(wr_nohold); Readln(i);
         writeln (wr_file, '(2) testing (single) : '); wr_flush(wr_nohold); Readln(r);
         writeln (wr_file, '(3) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(4) testing (integer): '); wr_flush(wr_nohold); Readln(i);
         writeln (wr_file, '(5) testing (single) : '); wr_flush(wr_nohold); Readln(r);
         writeln (wr_file, '(6) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(7) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(8) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(9) testing (string) : '); wr_flush(wr_nohold); Readln(s);

         writeln (wr_file, '(10) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(11) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(12) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(13) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(14) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(15) testing (string) : '); wr_flush(wr_nohold); Readln(s);

         writeln (wr_file, '(16) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(17) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(18) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(19) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(20) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(21) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(22) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         writeln (wr_file, '(23) testing (string) : '); wr_flush(wr_nohold); Readln(s);
         wr_flush(wr_nohold);
         wr_clear;
       END; {wr_test}


{ * * * * * * * * * * * * * * * * * * * * * * * *}


  BEGIN
  END. {Setts}
