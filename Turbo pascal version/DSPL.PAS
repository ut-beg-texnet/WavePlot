{RESOURCE: Display Resource}

UNIT Dspl;

  {Contains data structures & operators, needed to display the different
     graph types offered. This covers routines provided in the Graph Unit,
     procedures to perform functions (e.g. draw arrowheads), not provided
     in the unit, & windowing data structurs & operators}

 {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }


 INTERFACE

   USES
     Dsprim, Setts, Funct;


   {General, Images and Scaling}

     CONST
       ImageID  = $C53A;
       HeadSz   = 20;

     TYPE
       ImageHeader = RECORD
                       ID   : Word;
                       Ver  : Byte;
                       Mode : VidModeTypes;
                       Rbow : Boolean;
                       Dum1 : Byte;
                       Dum  : ARRAY [1..7] OF Word;
                     END;
       HelpType = (HelpMain,HelpGeom,HelpSnap,HelpDump,HelpHist,HelpSet);

     VAR
       G_DefExt, G_CurExt     : Extent;
       G_CurSc                : Single;
       SnapWind               : Ds_Window;
       Tdelay                 : Word;
       Vzoom                  : single;
       Scale2D                : RECORD
                                  {Data Set}
                                    Xaxis, Yaxis      : Char;
                                    Xgr, Ygr, Zgr     : Single;
                                    X1, X2, Y1, Y2    : Integer;
                                    Xst, Yst, Xstp, Ystp : Word; {Starting point and sample length in Data set}
                                  {Scale Set}
                                    TrueGscale           : Boolean;
                                    InvertVaxis          : Boolean;
                                    Dir                  : Word;      {0,1,2 - direction if 2D}
                                    XX1, XX2, YY1, YY2   : Integer;
                                    Xlen, Ylen           : Word;
                                    Xsz, Ysz             : Single;
                                    Xoff, Yoff           : LongInt;
                                    Magn                 : Integer;
                                    Xpx, Ypx             : LongInt;
                                END;

{       PROCEDURE TestPalette;
       PROCEDURE TestPalette2;}
     PROCEDURE Ds_ShowHelp (Context: Helptype);
     PROCEDURE Ds_SaveImage (Image_File, Image_Num, Image_path : String);
     FUNCTION  Ds_RestoreImage (Image_File: String) : Boolean;

     PROCEDURE Ds_Zoom (Magn: integer);
     PROCEDURE Ds_Shift (Xs,Ys: integer);
     PROCEDURE Ds_Enlarge (Xs,Ys: integer);
     PROCEDURE Set2Dscale;
     PROCEDURE Set3Dscale;
     PROCEDURE Ds_SnapScale (s: Word);
     PROCEDURE Ds_DumpScale (d: Word);
     PROCEDURE Ds_GeomScale (g, gDir : Word);
     PROCEDURE Ds_Titles;


   {Seismograms, 1-D graphing}

     TYPE
{       ScaleTypes = (ScLinked, ScNotLinked, ScHLinked, ScSet, ScSetLink);}
       ScaleTypes = (ScLinked, ScNotLinked, ScSet, ScSetLink);
     VAR
       Qtyn, Startn      : Single;
       Hzoom             : Single;
       {MxQty, MxLegMax   : Single;}
       Tmin, Tmax        : Single;
       HSst, HSfin       : Single;
       GroupVScale,
         GroupHScale     : ScaleTypes;
       LogDivV, LogDivH  : Integer;
       ScRound           : Integer;
       LineTypes         : Boolean;

     PROCEDURE Ds_InitSz;
     PROCEDURE Ds_ViewSeism (Choice : Char);
     PROCEDURE Ds_GetSeisms;
     PROCEDURE Ds_GraphSeism;


   {Geometry routines}
     VAR
       CurGeom, AxDir : Word;
       Rot            : Matrix3;
       Ang_x, Ang_y   : Integer;
     PROCEDURE Ds_Graph2dAxes;
     PROCEDURE Ds_Graph2DGeom (G, slice : Word);
     PROCEDURE Ds_Graph3DGeom (G : Word);
     PROCEDURE Ds_GeomLeg (G : Word);
     PROCEDURE Ds_GraphGeom (G : Word);
     PROCEDURE Draw_Traject (Loop:Integer; DelayOn, UnPlot: Boolean);
     PROCEDURE Draw_HistPos;
     PROCEDURE Ds_GraphCrackD (D,slice,CrackT : Word;  VAR ch:char);


   {Snapshot, 2-D graphing}

     CONST
       MaxCont   = 24;
       LineGzoom = 16;
     VAR
       CurSnap   : Word;
       Ds_Sc     : Single;

     FUNCTION  GetLevel (Value : Single): Word;
     PROCEDURE Ds_GraphContours;
     PROCEDURE Ds_LineGr;
     PROCEDURE QuickSnap;
     PROCEDURE InterpSnap;
     PROCEDURE FullSnap;
     PROCEDURE Ds_ListLevels;
     PROCEDURE Ds_GraphSnap;

     PROCEDURE DI_CvertAscii2 (MxMin: Integer);


   {Snapshot, 2-D Vector Plots - Stress/ Velocity}

     CONST
       VectZoom  = 4;
     VAR
       CurDump   : Word;

     PROCEDURE Ds_StressLeg   (Dmp : Word);
     PROCEDURE Ds_VectLeg     (Dmp : Word);
     PROCEDURE Ds_StressAxes  (Dmp : Word);
     PROCEDURE Ds_VelVects    (Dmp : Word);
     PROCEDURE Ds_DumpContour (Dmp : Word);
     PROCEDURE Ds_GraphDump;


  {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }


  IMPLEMENTATION

     USES
       DspForm, Graph, Crt, Data_Rsc, Plprim, Dos, ScrnBMP;


    {Redefine set_color command, to test for single-colour setting}
      PROCEDURE Ds_SelectColour (Col: Integer);
        VAR
          C: Word;
        BEGIN
           C := Col MOD 8;
           {Map so that 2nd col red, 3rd green}
           CASE C OF
             1  : Col := White;
             9  : Col := Yellow;
             2  : Col := LightRed;
             10 : Col := Red;
             3  : Col := LightBlue;
             11 : Col := Blue;
             4  : Col := LightGreen;
             12 : Col := Green;
             5  : Col := LightCyan;
             13 : Col := Cyan;
           END;
          IF (Sets.SameColour)
             THEN SetColor (DsCol_Fgrnd)
             ELSE SetColor (Col);
        END; {Ds_SelectColour}


   {Tempory Routines, not included later}
   {TestPalette, TestPalette2, Ds_Info}

{       PROCEDURE TestPalette;
         VAR
           Ch : Char;
           ix, iy : Word;
         BEGIN
           Ds_SetWindow (LegWind);
           FOR iy := 0 TO 15 DO
             FOR ix := 0 TO 15 DO
               BEGIN
                 SetFillStyle (SolidFill, iy*16+ix);
                 Bar (ix*9+2, iy*9+2, (ix+1)*9, (iy+1)*9);
               END;
           Ch := ReadKey;
           ClearViewPort;
           ResetViewPort;
         END; {TestPalette}

{       PROCEDURE TestPalette2;
         VAR
           Ch : Char;
           ix, iy : Word;
         BEGIN
           Ds_SetWindow (LegWind);
           FOR iy := 0 TO (Sets.QtyCols DIV 6) DO
             FOR ix := 0 TO 5 DO
               BEGIN
                 SetFillStyle (SolidFill, ColOffset+iy*6+ix);
                 Bar (ix*10+2, iy*10+2, (ix+1)*10, (iy+1)*10);
               END;
           Ch := ReadKey;
           ClearViewPort;
           ResetViewPort;
         END; {TestPalette2}

{     PROCEDURE Ds_Info (S : String; Col: Word; SNum: Word);
       VAR
         W     : ARRAY [1..2] OF Pixels;
         Vdiff : Integer;
         ColBak: Word;
       BEGIN
         SetTextStyle (SmallFont, HorizDir, FontSz);
         W[1,1] := Hscale DIV 8;
         W[1,2] := Vscale DIV 2 - 30 - Vscale DIV 5;
         W[2,1] := Hscale - W[1,1] - 50;
         W[2,2] := Vscale DIV 2 - 30 + Vscale DIV 5;
         Vdiff  := W[2,2] - W[1,2];
         ColBak := GetColor;
         SetColor  (Col);
         Rectangle   (W[1,1], W[1,2], W[2,1], W[2,2]);
         SetViewPort (W[1,1], W[1,2], W[2,1], W[2,2], ClipOn);
         IF (SNum = 2)
            THEN ClearViewPort;
         SetTextJustify (LeftText, CenterText);
         MoveTo ( 10, Vdiff DIV 10 * SNum + 4);
         OutText (s);
         ResetViewPort;
         SetColor (ColBak);
       END; {Ds_Info}


   {Routines for Image storage, compression and retrieval}
   {Ds_SaveImage, Ds_RestoreImage}

     FUNCTION Ds_ImageSize (x1,y1,x2,y2: Integer): Word;
       VAR
         xwid, ywid : Word;
       BEGIN
         CASE Sets.VidMode OF
{           std16, vga256 : Ds_ImageSize := ImageSize(x1,y1,x2,y2);}
           std16 :
                BEGIN
                  xwid:=x2-x1+1; ywid:=y2-y1+1; Ds_ImageSize:=((xwid SHR 1)+1)*ywid+4;
                END;
           vga256 :
                BEGIN
                  xwid:=x2-x1+1; ywid:=y2-y1+1; Ds_ImageSize:=xwid*ywid+4;
                END;
           ELSE BEGIN
                  xwid:=x2-x1+1; ywid:=y2-y1+1; Ds_ImageSize:=xwid*ywid+4;
                END;
         END; {Case}
       END; {Ds_ImageSize}

     PROCEDURE Ds_SaveImage (Image_File, Image_Num, Image_path : String);
       CONST
         Onebyte  = 1;
         MaxBlks  = 16;
         OrigBlks = 8;
       TYPE
         ImageBuf  = ARRAY [1..$8000] OF Byte;
       VAR
         QtyBlks         : Word;
         BufSz, Sz       : Word;
         Buffer          : ^ImageBuf;
         CrunchSz        : Word;
         Crunch          : ^ImageBuf;
         Hqty, Hst, Hfin : Word;
         i               : Word;
         DumFile         : Text;
         ImageFile       : File;
         Header          : ImageHeader;

       PROCEDURE Ds_Crunch (VAR CrunchSz : Word);
         CONST
           MaxCt : Byte = 240;
           Ext1  : Byte = 245;
           Ext2  : Byte = 253;
         VAR
           i, j        : LongInt;
           Count, Ch   : Byte;
         BEGIN
           j := 0;
           Ch := Buffer^[1];
           Count := 1;
           FOR i := 1 TO Sz DO
             IF ( (Buffer^[i+1] = Ch) AND (Count < MaxCt) AND (i<Sz) )
                THEN Inc (Count)
                ELSE
                  IF ( (Count <= 3) AND  (Ch <> Ext1) AND (Ch <> Ext2) )
                     THEN BEGIN
                            REPEAT
                              Inc (j);
                              Crunch^ [j] := Ch;
                              Dec (Count);
                            UNTIL (Count = 0);
                            Ch := Buffer^[i+1];
                            Count := 1;
                          END
                     ELSE
                       BEGIN
                         IF ( (Ch = Ext1) OR (Ch = Ext2) )
                            THEN BEGIN Inc(j); Crunch^[j] := Ext1; END;
                         Inc (j);
                         Crunch^[j] := Ch;
                         Inc(j);
                         Crunch^[j] := Ext2;
                         Inc(j);
                         Crunch^[j] := Count;
                         IF (i<Sz) THEN Ch := Buffer^[i+1];
                         Count := 1;
                       END;
           CrunchSz := j;
         END; {Ds_Crunch}

       BEGIN {Ds_SaveImage}
         GrafToBMP16(Image_path); {Snapshot of screen to BMP file}

         ResetViewPort;
         ASSIGN  (DumFile, (Image_File+'.I00'));
         REWRITE (DumFile); WRITE (DumFile, 'Image List: ');
         CLOSE   (DumFile);
         ASSIGN  (ImageFile, (Image_File+'.I'+Image_Num));
         REWRITE (ImageFile, OneByte);

         {Setup Header, to identify graphics mode and version}
           Header.ID   := ImageID;
           Header.Ver  := 1;
           Header.Mode := Sets.VidMode;
           Header.Rbow := Rainbow;
           BlockWrite (ImageFile, Header, HeadSz);

         IF (MaxAvail <= sizeof(imagebuf)) THEN exit;
         New (Crunch);
         IF (Sets.VidMode IN [svga1024])
            THEN QtyBlks := MaxBlks
            ELSE QtyBlks := OrigBlks;
         Hqty := (Hpix + QtyBlks - 1) DIV QtyBlks;
         BufSz := Ds_ImageSize (0,0,Hqty-1,Vpix-1);
         IF (MaxAvail > BufSz)
            THEN BEGIN
                   GetMem (Buffer, BufSz);
                   FOR i := 1 TO QtyBlks DO
                     BEGIN
                       Hst  := (i-1)*Hqty;
{                       IF (i=QtyBlks)
                          THEN Hfin := Hpix-1
                          ELSE Hfin := i*Hqty-1;}
                       IF (i<>QtyBlks)
                          THEN Hfin := i*Hqty-1
                          ELSE Hfin := Hpix-1;
                       Sz := Ds_ImageSize (Hst, 0, Hfin, Vpix-1);
                       GetImage  (Hst, 0, Hfin, Vpix-1, Buffer^);
                       IF (Sets.Compress)
                          THEN BEGIN
                                 Ds_Crunch (CrunchSz);
                                 BlockWrite (ImageFile, CrunchSz, 2);
                                 BlockWrite (ImageFile, Crunch^, CrunchSz);
                               END
                          ELSE BEGIN
                                 BlockWrite (ImageFile, Sz, 2);
                                 BlockWrite (ImageFile, Buffer^, Sz);
                               END;
                     END;
                   FreeMem (Buffer, BufSz);
                 END;
         Dispose (Crunch);
         CLOSE (ImageFile);
       END; {Ds_SaveImage}


     FUNCTION Ds_RestoreImage (Image_File: String) : Boolean;
       CONST
         Onebyte  = 1;
         MaxBlks  = 16;
         OrigBlks = 8;
       TYPE
         ImageBuf  = ARRAY [1..$8000] OF Byte;
       VAR
         QtyBlks         : Word;
         Sz              : ARRAY [1..MaxBlks] OF Word;
         Buffer          : ARRAY [1..MaxBlks] OF ^ImageBuf;
         CrunchSz        : Word;
         Crunch          : ^ImageBuf;
         Hqty, Hst, Hfin : Word;
         i,j             : Word;
         bl_st, bl_fin   : Word;
         HaveMem, Fail   : Boolean;
         DumFile         : Text;
         ImageFile       : File;
         Header          : ImageHeader;

       PROCEDURE Ds_Expand (Bf: Word);
         CONST
           MaxCt : Byte = 240;
           Ext1  : Byte = 245;
           Ext2  : Byte = 253;
         VAR
           i, j        : LongInt;
           Count, Ch   : Byte;
           tj          : ^Byte;
         BEGIN
           i := 0; j := 1;
           REPEAT
             IF (Crunch^[j]=Ext1) THEN Inc (j);
             Ch := Crunch^[j];
             Inc (j);
             IF ( (Crunch^[j] <> Ext2) OR (j > CrunchSz) )
                THEN BEGIN  Inc (i); Buffer[Bf]^[i] := Ch; END
                ELSE BEGIN
                       Inc(j);
                       FOR Count := 1 TO Crunch^[j] DO
                         BEGIN  Inc (i); Buffer[Bf]^[i] := Ch;  END;
                       Inc(j);
                     END;
           UNTIL (j > CrunchSz);
         END; {Ds_Expand}
(*
       PROCEDURE Ds_Expand (Bf: Word);
         CONST
           MaxCt : Byte = 240;
           Ext1  : Byte = 245;
           Ext2  : Byte = 253;
         VAR
           Count, Ch     : Byte;
           ti, tj, tjend : ^Byte;
         BEGIN
           ti := @Buffer[Bf]^[1]; tj := @Crunch^[1];
           tjend := @Crunch^[CrunchSz+1];
           REPEAT
             IF (tj^=Ext1) THEN Inc(tj);
             Ch := tj^;
             Inc (tj);
             IF ( (tj^ <> Ext2) OR (tj = tjend) )
                THEN BEGIN  ti^:=Ch; Inc(ti); END
                ELSE BEGIN
                       Inc(tj);
                       FOR Count := 1 TO tj^ DO
                         BEGIN  ti^ := Ch; Inc(ti);  END;
                       Inc(tj);
                     END;
           UNTIL (tj = tjend);
         END; {Ds_Expand} *)

       BEGIN {Ds_RestoreImage}
         ResetViewPort;
         ASSIGN  (ImageFile, Image_File);
         {$I-}
         RESET   (ImageFile, OneByte);
         {$I+}
         IF (IOResult <> 0)
            THEN Ds_RestoreImage := False
            ELSE BEGIN
                   BlockRead (ImageFile, Header, HeadSz);
                   IF (Header.ID <> ImageID)
                      THEN Seek (ImageFile, 0)
                      ELSE BEGIN
                             IF (Header.Mode <> Sets.VidMode)
                                THEN BEGIN
                                       CloseGraph;
                                       Sets.VidMode := Header.Mode;
                                       Ds_InitVid;
                                       Set_Colours;
                                       Ds_InitWindows;
                                       Ds_256Palette;
                                       IF (Header.Rbow)
                                          THEN Ds_SetPalette
                                          ELSE Ds_ResetPalette;
                                       SetBkColor (DsCol_Bgrnd);
                                       SetColor (DsCol_Fgrnd);
                                       Ds_DrawWindow (MsgWind);
                                       Ds_DrawWindow (LegWind);
                                       ResetViewPort;
                                     END;
{                             IF (Header.Rbow <> Rainbow)
                                THEN IF (Header.Rbow)
                                        THEN Ds_SetPalette
                                        ELSE Ds_ResetPalette;}
                           END;


                   New (Crunch);
                   IF (Sets.VidMode IN [svga1024])
                      THEN QtyBlks := MaxBlks
                      ELSE QtyBlks := OrigBlks;
                   Hqty := (Hpix+QtyBlks-1) DIV QtyBlks;
                   bl_st := 1; bl_fin :=0; Fail:=False;

                   FOR i := 1 TO (QtyBlks) DO
                     BEGIN
                       Hst  := (i-1)*Hqty;
                       IF (i=QtyBlks)
                          THEN Hfin := Hpix-1
                          ELSE Hfin := i*Hqty-1;
                       Sz[i] := Ds_ImageSize ( Hst, 0, Hfin, Vpix-1);

                      {load blocks into memory - when this fails display and
                        clear these blocks, then load the rest}
                       REPEAT

                         HaveMem := (MaxAvail >= Sz[i]);
                         IF (HaveMem)
                            THEN BEGIN
                                   GetMem (Buffer[i], Sz[i]);
                                   BlockRead (ImageFile, CrunchSz, 2);
                                   IF (CrunchSz = Sz[i])
                                      THEN BlockRead (ImageFile, Buffer[i]^, CrunchSz)
                                      ELSE BEGIN
                                             BlockRead (ImageFile, Crunch^, CrunchSz);
                                             Ds_Expand (i);
                                           END;
                                   bl_fin := i;
                                 END; {If}
                         IF (NOT(HaveMem) OR (i=QtyBlks))
                             THEN BEGIN
                                    FOR j := bl_st TO bl_fin DO
                                      IF (j=1)
                                         THEN PutImage ( 0, 0, Buffer[j]^, NormalPut)
                                         ELSE PutImage ( (j-1)*Hqty, 0, Buffer[j]^, NormalPut);

                                    FOR j := bl_st TO bl_fin DO
                                        FreeMem (Buffer[j], Sz[j]);
                                    Fail := (bl_fin < bl_st);
                                    bl_st := bl_fin+1;
                                  END; {If}

                       UNTIL (HaveMem OR Fail)

                     END; {For}

                   Dispose (Crunch);
                   CLOSE (ImageFile);
                   Ds_RestoreImage := NOT(Fail);

                 END; {Else}
       END; {Ds_RestoreImage}

   {$I help.pas}

   {Routines for Scaling}

     PROCEDURE Ds_Zoom (Magn: integer);
       VAR
         xd, yd : single;
         zoom   : single;
       BEGIN {Ds_Zoom}
         WITH Sets DO
           BEGIN
             GeomExt    := G_CurExt;
             GeomExtSet := True;
             WITH GeomExt DO
               BEGIN
                 zoom := exp(-Magn*0.1*ln(2));
                 xd  := (xp2-xp1) * zoom;
                 yd  := (yp2-yp1) * zoom;
                 xp1 := (xp1+xp2-xd)*0.5;
                 xp2 := xp1 + xd;
                 yp1 := (yp1+yp2-yd)*0.5;
                 yp2 := yp1 + yd;
               END;
           END; {with Sets}
       END; {Ds_Zoom}

     PROCEDURE Ds_Shift (Xs,Ys: integer);
       VAR
         xd, yd : single;
       BEGIN {Ds_Shift}
         WITH Sets DO
           BEGIN
             GeomExt    := G_CurExt;
             GeomExtSet := True;
             WITH GeomExt DO
               BEGIN
                 xd  := (xp2-xp1);
                 yd  := (yp2-yp1);
                 xp1 := xp1 + Xs*xd*0.01;
                 xp2 := xp2 + Xs*xd*0.01;
                 yp1 := yp1 + Ys*yd*0.01;
                 yp2 := yp2 + Ys*yd*0.01;
               END;
           END; {with Sets}
       END; {Ds_Shift}

     PROCEDURE Ds_Enlarge (Xs,Ys: integer);
       VAR
         xd, yd : single;
       BEGIN {Ds_Shift}
         WITH Sets DO
           BEGIN
             GeomExt    := G_CurExt;
             GeomExtSet := True;
             WITH GeomExt DO
               BEGIN
                 xd  := (xp2-xp1);
                 yd  := (yp2-yp1);
                 xp1 := xp1 - Xs*xd*0.01;
                 xp2 := xp2 + Xs*xd*0.01;
                 yp1 := yp1 - Ys*yd*0.01;
                 yp2 := yp2 + Ys*yd*0.01;
               END;
           END; {with Sets}
       END; {Ds_Enlarge}


     PROCEDURE Set2Dscale;
       CONST
         X0 = 1;
         Y0 = 1;
       VAR
         Margin  : Integer;
         MidX, MidY : Single;
         xrat, yrat : single;
       BEGIN {Set2Dscale}
         WITH Scale2D DO
           BEGIN
             {(a) Set default geom extents}
               WITH G_DefExt DO
                 BEGIN
                   margin := iMax((X2-X1), (Y2-Y1)) DIV 20;
                   IF (margin < 1) THEN margin:=1;
                   xp1 := (X1-X0-margin)*Xgr;  yp1:=(Y1-Y0-margin)*Ygr;
                   xp2 := (X2-X0+margin)*Xgr;  yp2:=(Y2-Y0+margin)*Ygr;
                 END;
             {(b) Set current extents. Depends on settings to default, or fixed, or scaled}
               IF NOT(Sets.GeomScSet)
                  THEN BEGIN
                         IF (Sets.GeomExtSet)
                            THEN G_CurExt := Sets.GeomExt
                            ELSE G_CurExt := G_DefExt;
                       END
                  ELSE IF (Sets.GeomExtSet)
                          THEN WITH Sets.GeomExt DO
                                 BEGIN
                                   MidX := (xp1+xp2)*0.5;
                                   MidY := (yp1+yp2)*0.5;
                                   G_CurExt.xp1 := MidX - PgLenHz*0.5 * Sets.GeomSc;
                                   G_CurExt.xp2 := MidX + PgLenHz*0.5 * Sets.GeomSc;
                                   G_CurExt.yp1 := MidY - PgLenVt*0.5 * Sets.GeomSc;
                                   G_CurExt.yp2 := MidY + PgLenVt*0.5 * Sets.GeomSc;
                                   xrat := (G_CurExt.xp2-G_CurExt.xp1);
                                   yrat := (G_CurExt.yp2-G_CurExt.yp1);
                                   IF (G_CurExt.xp1 < xp1) THEN G_CurExt.xp1 := xp1;
                                   IF (G_CurExt.xp2 > xp2) THEN G_CurExt.xp2 := xp2;
                                   IF (G_CurExt.yp1 < yp1) THEN G_CurExt.yp1 := yp1;
                                   IF (G_CurExt.yp2 > yp2) THEN G_CurExt.yp2 := yp2;
                                   xrat := (G_CurExt.xp2-G_CurExt.xp1) / xrat;
                                   yrat := (G_CurExt.yp2-G_CurExt.yp1) / yrat;
                                 END {With, Then}
                          ELSE WITH G_DefExt DO
                                 BEGIN
                                   MidX := (xp1+xp2)*0.5;
                                   MidY := (yp1+yp2)*0.5;
                                   G_CurExt.xp1 := MidX - PgLenHz*0.5 * Sets.GeomSc;
                                   G_CurExt.xp2 := MidX + PgLenHz*0.5 * Sets.GeomSc;
                                   G_CurExt.yp1 := MidY - PgLenVt*0.5 * Sets.GeomSc;
                                   G_CurExt.yp2 := MidY + PgLenVt*0.5 * Sets.GeomSc;
                                 END; {With, else}
             {(c) Calculate the new (current) scale}
               IF (Sets.GeomScSet)
                  THEN G_CurSc := Sets.GeomSc
                  ELSE WITH G_CurExt DO
                         G_CurSc := (xp2-xp1)/PgLenHz;
           END; {With Scale2D}

         WITH Scale2D DO
           BEGIN
             {(a) Set other scaling values, using current extents}
               WITH G_CurExt DO
                 BEGIN
                   XX1 := X0+TRUNC(xp1/Xgr-0.4999); XX2 := X0+TRUNC(xp2/Xgr+0.4999);
                   YY1 := Y0+TRUNC(yp1/Ygr-0.4999); YY2 := Y0+TRUNC(yp2/Ygr+0.4999);
(*                   xp1 := (XX1-X0)*Xgr; xp2 := (XX2-X0)*Xgr;
                   yp1 := (YY1-Y0)*Ygr; yp2 := (YY2-Y0)*Ygr;*)
                   Xsz := (xp2-xp1);   Ysz := (yp2-yp1);
                   Xlen := XX2-XX1;    Ylen := YY2-YY1;
                 END;
             {(b) Set pixel scales and offsets}
               Xpx := SnapWindMx.Sz[1]-1;
               Ypx := SnapWindMx.Sz[2]-1;
               IF NOT(TrueGscale)
                  THEN BEGIN  Xoff:=0; Yoff:=0;  END
                  ELSE BEGIN
                         IF (Sets.GeomScSet)
                            THEN BEGIN  Xpx := TRUNC(Xpx*xrat); Ypx := TRUNC(Ypx*yrat);  END;
                         IF ((Xpx/Xsz) > (Ypx/Ysz))
                            THEN Xpx := TRUNC (Ypx * Xsz / Ysz)
                            ELSE Ypx := TRUNC (Xpx * Ysz / Xsz);
                         Xoff := (SnapWindMx.Sz[1]-1 - Xpx) DIV 2;
                         Yoff := (SnapWindMx.Sz[2]-1 - Ypx) DIV 2;
                       END; {else}
             {(c) Set snapshot window}
               WITH SnapWind DO
                WITH G_CurExt DO
                 BEGIN
                   St[1]  := SnapWindMx.St[1] + Xoff;
                   St[2]  := SnapWindMx.St[2] + Yoff;
                   Fin[1] := SnapWindMx.Fin[1] - Xoff;
                   Fin[2] := SnapWindMx.Fin[2] - Yoff;
                   Sz[1]  := Fin[1] - St[1] + 1;
                   Sz[2]  := Fin[2] - St[2] + 1;
                  {Set the offset of the graph relative to current extent/border
                     and adjust the total number of pixels for scaling.}
            {These SHOULD ALL BE ZERO?????}
(*                   Xoff := TRUNC (Xpx*((XX1-X0)*Xgr-xp1) /Xsz) DIV 2;
                   Xpx  := Sz[1]-1-Xoff*2 - TRUNC(Xpx*(xp2-(XX2-X0)*Xgr) /Xsz);
                   Yoff := TRUNC (Ypx*((YY1-Y0)*Ygr-yp1) /Ysz) DIV 2;
                   Ypx  := Sz[2]-1-Yoff*2 - TRUNC(Ypx*(yp2-(YY2-Y0)*Ygr) /Ysz);*)
{                   Yoff := TRUNC (Ypx*(yp2-(YY2-Y0)*Ygr) /Ysz) DIV 2;
                   Ypx  := Sz[2]-1-Yoff*2 - TRUNC(Ypx*((YY1-Y0)*Ygr-yp1) /Ysz);}
                   Xoff := 0; Xpx:=Sz[1]-1;
                   Yoff := 0; Ypx:=Sz[2]-1;
                   IF (Yaxis IN ['j','J'])  {Need to invert - i.e. j1 at bottom, j2 at top}
                      THEN BEGIN  Yoff := Yoff + Ypx;  Ypx  := -Ypx;  END;
                 END; {With SnapWind and G_CurExt}
             {(d) Set direction}
               IF (TrueGscale) THEN
                  BEGIN
                   IF ((Xaxis='i') AND (Yaxis='j')) THEN Dir:=0;
                   IF ((Xaxis='i') AND (Yaxis='k')) THEN Dir:=1;
                   IF ((Xaxis='k') AND (Yaxis='j')) THEN Dir:=2;
                  END; {If}
           END; {With Scale2D}
       END; {Set2Dscale}


     PROCEDURE Set3Dscale;
       CONST
         X0 = 1;
         Y0 = 1;
         Z0 = 1;
       VAR
         Margin  : Integer;
         Mgr     : single;
       BEGIN {Set3Dscale}
         WITH Scale2D DO
           BEGIN
             {(a) Set default geom extents}
               WITH G_DefExt DO
                 BEGIN
                   margin := iMax((X2-X1), (Y2-Y1)) DIV 20;
                   IF (margin < 1) THEN margin:=1;
                   xp1 := (X1-X0-margin)*Xgr;  yp1:=(Y1-Y0-margin)*Ygr;
                   xp2 := (X2-X0+margin)*Xgr;  yp2:=(Y2-Y0+margin)*Ygr;
                 END;
             {(b) Set current extents. Only default, or fixed - no scaling allowed}
                    {- i.e. ignores Sets.GeomScSet}
               IF (Sets.GeomExtSet)
                  THEN G_CurExt := Sets.GeomExt
                  ELSE G_CurExt := G_DefExt;
             {(c) Calculate the new (current) scale - not true scale}
               WITH G_CurExt DO
                 G_CurSc := (xp2-xp1)/PgLenHz;
           END; {With Scale2D}

         WITH Scale2D DO
           BEGIN
             {(a) Set other scaling values, using current extents}
                  {Used Xgr, compensate for different Ygr and Zgr in Rotation matrix}
               WITH G_CurExt DO
                 BEGIN
                   Mgr := Max(Max(Xgr,Ygr),Zgr);
                   XX1 := X0+TRUNC(xp1/Mgr-0.4999); XX2 := X0+TRUNC(xp2/Mgr+0.4999);
                   YY1 := Y0+TRUNC(yp1/Mgr-0.4999); YY2 := Y0+TRUNC(yp2/Mgr+0.4999);
                   Xsz := (xp2-xp1);        Ysz := (yp2-yp1);
                   Xlen := XX2-XX1;         Ylen := YY2-YY1;
                   Xlen := ROUND (Xlen*3);  Ylen := ROUND (Ylen*3); {reduce scale}
                 END;
             {(b) Set pixel scales and offsets}
               Xpx := SnapWindMx.Sz[1]-1;
               Ypx := SnapWindMx.Sz[2]-1;
               IF ((Xpx/Xsz) > (Ypx/Ysz))
                  THEN Xpx := TRUNC (Ypx * Xsz / Ysz)
                  ELSE Ypx := TRUNC (Xpx * Ysz / Xsz);
               Xoff := (SnapWindMx.Sz[1]-1) DIV 2;
               Yoff := (SnapWindMx.Sz[2]-1) DIV 2;
               Ypx  := -Ypx;
             {(c) Set snapshot window}
             SnapWind:=SnapWindMx;
             {Set rotation matrix}
             Mat_I3 (Rot);
             Rot[1,1]:=Xgr/Mgr; Rot[2,2]:=Ygr/Mgr; Rot[3,3]:=Zgr/Mgr;
             Mat_Rot3D (Rot,1,Ang_x);
             Mat_Rot3D (Rot,2,Ang_y);
           END; {With Scale2D}
       END; {Set3Dscale}


     PROCEDURE Ds_SnapScale (s: Word);
       BEGIN
         WITH Scale2D DO
         WITH SnapMap^ [s] DO
{         With CurSnapRec DO}
           BEGIN
             TrueGscale := ([Ax1, Ax2] * ['T', 't'] = []);
             Xstp := sx;   Ystp := sy;
             Xgr  := dx;   Ygr  := dy;
             Xaxis := Ax1; Yaxis := Ax2;
             CASE Ax1 OF
               'i', 'I' : BEGIN X1 := i1; X2 := i2; Xst := ( (i2-i1) MOD sx) DIV 2 + 1; END;
               'j', 'J' : BEGIN X1 := j1; X2 := j2; Xst := ( (j2-j1) MOD sx) DIV 2 + 1; END;
               'k', 'K' : BEGIN X1 := k1; X2 := k2; Xst := ( (k2-k1) MOD sx) DIV 2 + 1; END;
               't', 'T' : BEGIN X1 := ROUND(t1/dx);  X2 :=ROUND(time/dx);  Xst := sx;   END;
             END;
             CASE Ax2 OF
               'i', 'I' : BEGIN Y1 := i1; Y2 := i2; Yst := ( (i2-i1) MOD sy) DIV 2 + 1; END;
               'j', 'J' : BEGIN Y1 := j1; Y2 := j2; Yst := ( (j2-j1) MOD sy) DIV 2 + 1; END;
               'k', 'K' : BEGIN Y1 := k1; Y2 := k2; Yst := ( (k2-k1) MOD sy) DIV 2 + 1; END;
               't', 'T' : BEGIN Y1 := ROUND(t1/dy);  Y2 :=ROUND(time/dy);  Yst := sy;  END;
             END;
             Slice2D := 0;
             IF (TrueGscale AND SliceG) THEN
                BEGIN
                 IF ((Ax1='i') AND (Ax2='j')) THEN Slice2D:=k1;
                 IF ((Ax1='i') AND (Ax2='k')) THEN Slice2D:=j1;
                 IF ((Ax1='k') AND (Ax2='j')) THEN Slice2D:=i1;
                END; {If}
           END;
         Set2Dscale;
       END; {Ds_SnapScale}


     PROCEDURE Ds_DumpScale (d: Word);
       BEGIN
         WITH Scale2D DO
         With DumpMap^ [d] DO
           BEGIN
             IF ((kqty>1) AND (jqty>1))
                THEN BEGIN  Xqty:=kqty; Ax1:='k'; Xstp := sz; Xgr  := dz;  END
                ELSE BEGIN  Xqty:=iqty; Ax1:='i'; Xstp := sx; Xgr  := dx;  END;
             IF (jqty > 1)
                THEN BEGIN  Yqty:=jqty; Ax2:='j'; Ystp := sy; Ygr  := dy;  END
                ELSE BEGIN  Yqty:=kqty; Ax2:='k'; Ystp := sz; Ygr  := dz;  END;
             TrueGscale := True;
             Xaxis := Ax1; Yaxis := Ax2;
             CASE Ax1 OF
               'i', 'I' : BEGIN X1 := i1; X2 := i2; Xst := ( (i2-i1) MOD Xstp) DIV 2 + 1; END;
               'j', 'J' : BEGIN X1 := j1; X2 := j2; Xst := ( (j2-j1) MOD Xstp) DIV 2 + 1; END;
               'k', 'K' : BEGIN X1 := k1; X2 := k2; Xst := ( (k2-k1) MOD Xstp) DIV 2 + 1; END;
             END;
             CASE Ax2 OF
               'i', 'I' : BEGIN Y1 := i1; Y2 := i2; Yst := ( (i2-i1) MOD Ystp) DIV 2 + 1; END;
               'j', 'J' : BEGIN Y1 := j1; Y2 := j2; Yst := ( (j2-j1) MOD Ystp) DIV 2 + 1; END;
               'k', 'K' : BEGIN Y1 := k1; Y2 := k2; Yst := ( (k2-k1) MOD Ystp) DIV 2 + 1; END;
             END;
             Slice2D := 0;
             IF (TrueGscale AND SliceG) THEN
                BEGIN
                 IF ((Ax1='i') AND (Ax2='j')) THEN Slice2D:=k1;
                 IF ((Ax1='i') AND (Ax2='k')) THEN Slice2D:=j1;
                 IF ((Ax1='k') AND (Ax2='j')) THEN Slice2D:=i1;
                END; {If}
           END;
         Set2Dscale;
       END; {Ds_DumpScale}

     PROCEDURE Ds_GeomScale (g, gDir : Word);
       BEGIN
         WITH Scale2D DO
         With GeomMap^ [g] DO
           BEGIN
             TrueGscale := true;
             Dir := gDir;
             IF (Sets.Set3D) THEN gDir := 0;
             CASE gDir OF
               0 : BEGIN Xaxis := 'i'; Yaxis := 'j'; END;
               1 : BEGIN Xaxis := 'i'; Yaxis := 'k'; END;
               2 : BEGIN Xaxis := 'k'; Yaxis := 'j'; END;
             END; {Case}
             Xstp := 1; Ystp := 1;
             CASE Xaxis OF
               'i', 'I' : BEGIN X1 := i1; X2 := i2; Xgr := dx;  END;
               'j', 'J' : BEGIN X1 := j1; X2 := j2; Xgr := dy;  END;
               'k', 'K' : BEGIN X1 := k1; X2 := k2; Xgr := dz;  END;
             END;
             CASE Yaxis OF
               'i', 'I' : BEGIN Y1 := i1; Y2 := i2; Ygr := dx;  END;
               'j', 'J' : BEGIN Y1 := j1; Y2 := j2; Ygr := dy;  END;
               'k', 'K' : BEGIN Y1 := k1; Y2 := k2; Ygr := dz;  END;
             END;
             IF (Sets.Set3D) THEN Zgr:=dz;
             Xst := 1; Yst := 1;
             IF (Sets.Set3D)
                THEN Set3Dscale
                ELSE Set2Dscale;
           END; {With GeomMap^, Scale2D}
       END; {Ds_GeomScale}


     PROCEDURE Ds_Titles;
       BEGIN
         Ds_SelectColour (DsCol_Titles);
         Ds_SetWindow (HdTwind);
         SetTextStyle (SmallFont, HorizDir, FontSz);
         SetTextJustify (CenterText, CenterText);
         OutTextXY (HdTwind.Sz[1] DIV 2, HdTwind.Sz[2] DIV 2, Sets.HdTitle);
         Ds_SetWindow (HzTwind);
         SetTextStyle (SmallFont, HorizDir, FontSz);
         SetTextJustify (CenterText, CenterText);
         OutTextXY (HzTwind.Sz[1] DIV 2, HzTwind.Sz[2] DIV 2, Sets.HzTitle);
         Ds_SetWindow (VtTwind);
         SetTextStyle (SmallFont, VertDir, FontSz);
         SetTextJustify (CenterText, CenterText);
         OutTextXY (VtTwind.Sz[1] DIV 2, VtTwind.Sz[2] DIV 2, Sets.VtTitle);
       END; {Ds_Titles}

     {Seismogram Routines}
     {$I Dsp\DspSeism}

     {Geometry and grid Routines}
     {$I Dsp\DspGeom}

     {Snap Routines}
     {$I Dsp\DspSnap}

     {Dump Routines}
     {$I Dsp\DspDump}

   END. {Dspl Unit}
