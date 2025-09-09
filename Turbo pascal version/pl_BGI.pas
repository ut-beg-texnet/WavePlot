{RESOURCE: PLOTTER Resource}

  UNIT Pl_BGI;

   {Produces plotter output using RD-GL / HP-GL graphic language constructs}
   {Contains Constants, Data Structures & operators, useful for accessing the
    plotter. It implements the same operations provided for screen graphics
    by the Display Resource and BGI graphics unit. It maintains a Viewport
    and the operations provided are ViewPort relative.}


   {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *}

   INTERFACE


       USES
         Dos, Funct, Graph;

       TYPE
         PlSizes     = ARRAY [1..2] OF Integer;
         Pl_Point    = PlSizes;
         Pl_Win      = RECORD
                         St, Fin, Sz : PlSizes;
                       End;
       CONST
         Pl_1cm     = 400;    {The scale when in HPGL mode is 400 = 1cm of physical length}
         SheetSizes : PlSizes  = (10800,7600);

       VAR
         Pl_Viewport: Pl_Win;      {Current View of the Plot}
         Pl_CP      : Pl_Point;    {Current Pointer in Plot Space}

         PlotAsp    : Single;
         PlotSize   : PlSizes;

         Pl_Yaxis, Pl_Zaxis : Integer;  {Indicate along which plot axes, the Y & Z axes are.}


     {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *}

       CONST {Plotter Character Commands}
         Sep           = ',';
         Pl_Coords     = 'IP';
         Pl_Home       = 'PA PU 0,0';
         Pl_Move       = 'PU PA';
         Pl_MvRel      = 'PU PR';
         Pl_Draw       = 'PD PA';
         Pl_DrawRel    = 'PD PA';
{         Pl_Move       = 'PA PU';
         Pl_MvRel      = 'PR PU';
         Pl_Draw       = 'PA PD';
         Pl_DrawRel    = 'PR PD';}
         Pl_PenChange  = 'SP';
         Pl_LineType   = 'LT';
         Pl_PenThick   = 'PT';
         Pl_Rect       = 'EA';
         Pl_Circ       = 'CI';
         Pl_InpWindow  = 'IW';
         Pl_SetScale   = 'SC';
         Pl_Label      = 'LB';
         Pl_EndLabel   = CHR(3);
         Pl_CharSize   = 'SI';
         Pl_CharRsize  = 'SR';
         Pl_CharDir    = 'DI';
         Pl_CharSpace  = 'CP';
         Pl_Page       = 'PG';

       VAR
         Pl                 : Text;
         Pl_Hjust, Pl_Vjust : Word;  {Text justification Settings}
         Pl_TxtDir          : Word;  {Direction of Text}


     {Initialization of UNIT}

       PROCEDURE Pl_SetGlobalCoord;


     {Viewport control operations}

       PROCEDURE Pl_ResetViewPort;

       PROCEDURE Pl_SetViewPort (Hz_st, Vt_st, Hz_end, Vt_end : Integer);

       FUNCTION Pl_InView (P: Pl_Point): Boolean;

       FUNCTION Pl_Clip (VAR  P1, P2: Pl_Point): Boolean;


     {Plotting Primitives - equivalence to those provided for screen in Graphics Unit}

       PROCEDURE Pl_SetColor (Pen : Integer);

       PROCEDURE Pl_Line (Hz_st, Vt_st, Hz_end, Vt_end : Integer);

       PROCEDURE Pl_LineTo (Hz_end, Vt_end : Integer);

       PROCEDURE Pl_LineRel (Hz_incr, Vt_incr : Integer);

       PROCEDURE Pl_MoveTo (Hz_end, Vt_end : Integer);

       PROCEDURE Pl_MoveRel (Hz_incr, Vt_incr : Integer);

       PROCEDURE Pl_SetLineStyle (LineStyle, Pattern, Thickness: Word);

       PROCEDURE Pl_Rectangle (Hz_st, Vt_st, Hz_end, Vt_end : Integer);

       PROCEDURE Pl_Circle (Hz_st, Vt_st, Rad : Integer);

       PROCEDURE Pl_SetTextJustify (Hjust, Vjust: Word);

       PROCEDURE Pl_OutText (s: String);

       PROCEDURE Pl_OutTextXY (Hz_st, Vt_st: Integer; s: String);

       PROCEDURE Pl_SetTextStyle (Font, Direction, FontMagn: Word);


     {Plotting - File Operators}

       PROCEDURE Pl_InitPlot;

       PROCEDURE Pl_EndPlot (Qty : Word);

       PROCEDURE Pl_SavePlot (PathName : String);


    {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *}

   IMPLEMENTATION


     {Initialization of UNIT}

       PROCEDURE Pl_SetGlobalCoord;
         BEGIN
           Writeln (Pl, Pl_Coords);
           Writeln (Pl, Pl_Coords, 0, Sep, SheetSizes[2], Sep, SheetSizes[1], Sep, 0);
         END; {Pl_SetGlobalCoord}


     {Viewport control operations}

       PROCEDURE Pl_ResetViewPort;
         CONST
           CP_Init : Pl_Point = (0,0);
         BEGIN
           Pl_ViewPort.St[1]  := 0;
           Pl_ViewPort.St[2]  := 0;
           Pl_ViewPort.Sz[1]  := SheetSizes[1];
           Pl_ViewPort.Sz[2]  := SheetSizes[2];
           Pl_ViewPort.Fin[1] := SheetSizes[1]-1;
           Pl_ViewPort.Fin[2] := SheetSizes[2]-1;
           Pl_CP := CP_Init;
         END; {Pl_ResetViewPort}

       PROCEDURE Pl_SetViewPort (Hz_st, Vt_st, Hz_end, Vt_end : Integer);
         CONST
           CP_Init : Pl_Point = (0,0);
         BEGIN
           Pl_ViewPort.St[1]  := Hz_st;
           Pl_ViewPort.St[2]  := Vt_st;
           Pl_ViewPort.Fin[1] := Hz_end;
           Pl_ViewPort.Fin[2] := Vt_end;
           Pl_ViewPort.Sz[1]  := Hz_end - Hz_st;
           Pl_ViewPort.Sz[2]  := Vt_end - Vt_st;
           Pl_CP := CP_Init;
         END; {Pl_SetViewPort}

       FUNCTION Pl_InView (P: Pl_Point): Boolean;
         BEGIN
           IF ( (P[1] <= Pl_ViewPort.Sz[1]) AND  (P[2] <= Pl_ViewPort.Sz[2]) AND
                (P[1] >= 0) AND (P[2] >= 0) )
              THEN Pl_InView := True
              ELSE Pl_InView := False;
         END; {Pl_InView}


       FUNCTION Pl_Clip (VAR  P1, P2: Pl_Point): Boolean;
         VAR
           Cut     : ARRAY [1..2] OF Pl_Point;
           QtyCuts : Integer;

         FUNCTION Pl_AlongLine (P : Pl_Point): Boolean;
           BEGIN
             IF (  (  ( (P1[1] <= P[1]) AND (P2[1] >= P[1]) ) OR ( (P2[1] <= P[1]) AND (P1[1] >= P[1]) )  )  AND
                   (  ( (P1[2] <= P[2]) AND (P2[2] >= P[2]) ) OR ( (P2[2] <= P[2]) AND (P1[2] >= P[2]) )  )  )
                THEN Pl_AlongLine := True
                ELSE Pl_AlongLine := False;
           END; {Pl_AlongLine}

         PROCEDURE  FindIntercepts;
           {Finds where a line intercepts with the current plotter viewport}
           {The strategy used is to extend the line and the four sides of the viewport, to infinity. The
            intersections of the line with each side can then be found. Each of the four intersections thus
            obtained is then tested and discarded if it does not lie along the line segment, or in the viewport.
            0 to 2 true intersections are obtained in this manner.}
           VAR
             XCut   : Pl_Point;   {Extended cut - obtained by extending line & viewport boundaries to infinity.}
             Diff   : PlSizes;
             i,j    : Integer;
           BEGIN
             QtyCuts := 0;
             Diff[1] := P2[1] - P1[1];
             Diff[2] := P2[2] - P1[2];
             FOR i := 1 TO 2 DO       {Horizontal & Vertical sides of the viewport}
               FOR j := 0 TO 1 DO     {2 intercepts for each}
                 BEGIN
                   XCut[3-i] := j * Pl_ViewPort.Sz[3-i]; {Take the known value for one of the Hz/Vt sides.}
                   IF (Diff[3-i] = 0)
                      THEN XCut[i] := P1[i]
                      ELSE XCut[i] := IROUND ( P1[i] + (XCut[3-i] - P1[3-i]) * (Diff[i] / Diff[3-i]) );
                           {Yintc = Yline + (Zintc-Zline) * YZslope  OR  Zintc = Zline + (Yintc-Yline) * ZYslope}
                   IF ( Pl_InView(XCut) AND Pl_AlongLine(XCut) )
                      THEN BEGIN
                             INC (QtyCuts); {It is a true intercept}
                             Cut [QtyCuts] := XCut;
                           END;
                 END; {For j}
           END; {FindIntercepts}

         BEGIN {Pl_Clip}
           FindIntercepts;
           CASE QtyCuts OF
              0 :   Pl_Clip := False;
              1 :   BEGIN
                      IF NOT (Pl_InView(P1))
                         THEN P1 := Cut[1];
                      IF NOT (Pl_InView(P2))
                         THEN P2 := Cut[1];
                      Pl_Clip := True;
                    END; {Case 1}
              2 :   BEGIN
                      P1 := Cut[1];
                      P2 := Cut[2];
                      Pl_Clip := True;
                    END; {Case 2}
           END; {Case QtyCuts}
         END; {Pl_Clip}


     {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *}
     {Plotting Primitives - equivalence to those provided for screen in Graphics Unit}

       PROCEDURE Pl_SetColor (Pen : Integer);
         BEGIN
           Writeln (Pl, Pl_PenChange, Sep, Pen);
         END; {Pl_SetColor}

       PROCEDURE Pl_Line (Hz_st, Vt_st, Hz_end, Vt_end : Integer);
         VAR
           In_Port : Boolean;
           P1, P2  : Pl_Point;
         BEGIN
           {Pl_CP - Current Pointer is not updated, except by relative commands.}
           P1[1] := Hz_st;   P1[2] := Vt_st;
           P2[1] := Hz_end;  P2[2] := Vt_end;
           IF ( Pl_InView(P1) AND Pl_InView(P2) )
              THEN In_Port := True
              ELSE In_Port := Pl_Clip (P1, P2);
           IF In_Port
              THEN BEGIN
                     Writeln (Pl, Pl_Move, (P1[1] + Pl_ViewPort.St[1]), Sep, (P1[2] + Pl_ViewPort.St[2]) );
                     Writeln (Pl, Pl_Draw, (P2[1] + Pl_ViewPort.St[1]), Sep, (P2[2] + Pl_ViewPort.St[2]) );
                   END;
         END; {Pl_Line}

       PROCEDURE Pl_LineTo (Hz_end, Vt_end : Integer);
         VAR
           P2   : Pl_Point;
         BEGIN
           P2[1] := Hz_end;  P2[2] := Vt_end;
           IF ( Pl_InView(Pl_CP) AND Pl_InView(P2) )
              THEN Writeln (Pl, Pl_Draw, (P2[1] + Pl_ViewPort.St[1]), Sep, (P2[2] + Pl_ViewPort.St[2]) )
              ELSE IF ( Pl_Clip (Pl_CP, P2) )
                      THEN BEGIN
                             Writeln (Pl, Pl_Move, (Pl_CP[1] + Pl_ViewPort.St[1]), Sep, (Pl_CP[2] + Pl_ViewPort.St[2]) );
                             Writeln (Pl, Pl_Draw, (P2[1] + Pl_ViewPort.St[1]), Sep, (P2[2] + Pl_ViewPort.St[2]) );
                           END;
           Pl_CP[1] := Hz_end;  Pl_CP[2] := Vt_end;  {The Current Pointer is never clipped}
         END; {Pl_LineTo}

       PROCEDURE Pl_LineRel (Hz_incr, Vt_incr : Integer);
         VAR
           P1,P2 : Pl_Point;
         BEGIN
           P1    := Pl_CP;
           P2[1] := Pl_CP[1] + Hz_incr;
           P2[2] := Pl_CP[2] + Vt_incr;
           Pl_CP := P2;   {Current Pointer must not be clipped}
           IF ( Pl_InView(P1) AND Pl_InView(P2) )
              THEN Writeln (Pl, Pl_DrawRel, Hz_incr, Sep, Vt_incr)
              ELSE IF ( Pl_Clip (P1, P2) )
                      THEN BEGIN
                             Writeln (Pl, Pl_Move, (P1[1] + Pl_ViewPort.St[1]), Sep, (P1[2] + Pl_ViewPort.St[2]) );
                             Writeln (Pl, Pl_Draw, (P2[1] + Pl_ViewPort.St[1]), Sep, (P2[2] + Pl_ViewPort.St[2]) );
                           END;
         END; {Pl_LineRel}

       PROCEDURE Pl_MoveTo (Hz_end, Vt_end : Integer);
         BEGIN
           Pl_CP[1] := Hz_end;
           Pl_CP[2] := Vt_end;
           Writeln (Pl, Pl_Move, (Hz_end + Pl_ViewPort.St[1]), Sep, (Vt_end + Pl_ViewPort.St[2]) );
         END; {Pl_MoveTo}

       PROCEDURE Pl_MoveRel (Hz_incr, Vt_incr : Integer);
         BEGIN
           Pl_CP[1] := Pl_CP[1] + Hz_incr;
           Pl_CP[2] := Pl_CP[2] + Vt_incr;
           Writeln (Pl, Pl_MvRel, Hz_incr, Sep, Vt_incr);
         END; {Pl_MoveTo}

       PROCEDURE Pl_SetLineStyle (LineStyle, Pattern, Thickness: Word);
         CONST
           SolidLn  = 0;
           DashedLn = 1;
         BEGIN
           CASE LineStyle OF
              DashedLn: Writeln (Pl, Pl_LineType, 2, Sep, '0.5');
              SolidLn:  Writeln (Pl, Pl_LineType,';');
              ELSE Writeln (Pl, Pl_LineType,';');
           END; {Case}
         END; {Pl_SetLineStyle}

       PROCEDURE Pl_Rectangle (Hz_st, Vt_st, Hz_end, Vt_end : Integer);
         VAR
           In_Port : Boolean;
           P1, P2  : Pl_Point;
         BEGIN
           {Pl_CP - Current Pointer is not updated, except by relative commands.}
{           P1[1] := Hz_st;  P1[2] := Vt_st;
           P2[1] := Hz_end; P2[2] := Vt_end;
           IF ( Pl_InView(P1) AND Pl_InView(P2) )
              THEN In_Port := True
              ELSE In_Port := Pl_Clip (P1, P2);}
{           IF In_Port
              THEN BEGIN
                     Writeln (Pl, Pl_Move, (P1[1] + Pl_ViewPort.St[1]), Sep, (P1[2] + Pl_ViewPort.St[2]) );
                     Writeln (Pl, Pl_Draw, (P2[1] + Pl_ViewPort.St[1]), Sep, (P1[2] + Pl_ViewPort.St[2]) );
                     Writeln (Pl, Pl_Draw, (P2[1] + Pl_ViewPort.St[1]), Sep, (P2[2] + Pl_ViewPort.St[2]) );
                     Writeln (Pl, Pl_Draw, (P1[1] + Pl_ViewPort.St[1]), Sep, (P2[2] + Pl_ViewPort.St[2]) );
                     Writeln (Pl, Pl_Draw, (P1[1] + Pl_ViewPort.St[1]), Sep, (P1[2] + Pl_ViewPort.St[2]) );
                   END;}
           Pl_Line (Hz_st,  Vt_st,  Hz_st,  Vt_end);  {Left-Bottom, Up}
           Pl_Line (Hz_st,  Vt_end, Hz_end, Vt_end);  {Right}
           Pl_Line (Hz_end, Vt_end, Hz_end, Vt_st );  {Down}
           Pl_Line (Hz_end, Vt_st,  Hz_st,  Vt_st );  {Left}
         END; {Pl_Rectangle}

       PROCEDURE Pl_Circle (Hz_st, Vt_st, Rad : Integer);
         BEGIN
           Writeln (Pl, Pl_Move, (Hz_st + Pl_ViewPort.St[1]), Sep, (Vt_st + Pl_ViewPort.St[2]) );
           Writeln (Pl, Pl_Circ, Rad);
         END; {Pl_Circle}

       PROCEDURE Pl_SetTextJustify (Hjust, Vjust: Word);
         BEGIN
           Pl_Hjust := Hjust;
           Pl_Vjust := Vjust;
         END; {Pl_SetTextJustify}

       PROCEDURE Pl_OutText (s: String);
         BEGIN
           IF (Pl_TxtDir = HorizDir)
              THEN Writeln (Pl, Pl_CharSpace, - (Length(s) * Pl_Hjust *0.5 *0.8) :5:2, Sep, -(Pl_Vjust/2/2):5:2 )
              ELSE Writeln (Pl, Pl_CharSpace, - (Length(s) * Pl_Hjust *0.5 *0.8) :5:2, Sep, -(Pl_Vjust/2/2):5:2 );
           Writeln (Pl, Pl_Label, s, Pl_Endlabel);
         END; {Pl_OutText}

       PROCEDURE Pl_OutTextXY (Hz_st, Vt_st: Integer; s: String);
         BEGIN
           Pl_CP[1] := Hz_st;
           Pl_CP[2] := Vt_st;
           Writeln (Pl, Pl_Move, (Hz_st + Pl_ViewPort.St[1]), Sep, (Vt_st + Pl_ViewPort.St[2]) );
           IF (Pl_TxtDir = HorizDir)
              THEN Writeln (Pl, Pl_CharSpace, - (Length(s) * Pl_Hjust *0.5*0.8) :5:2, Sep, -(Pl_Vjust/2/2):5:2 )
              ELSE Writeln (Pl, Pl_CharSpace, - (Length(s) * Pl_Hjust *0.5*0.8) :5:2, Sep, -(Pl_Vjust/2/2):5:2 );
           Writeln (Pl, Pl_Label, s, Pl_Endlabel);
         END; {Pl_OutTextXY}

       PROCEDURE Pl_SetTextStyle (Font, Direction, FontMagn: Word);
         VAR
           FontMult, FontDiv : Word;
           RelSize : Single;
         BEGIN
           Pl_TxtDir := Direction;
           IF (Pl_TxtDir = HorizDir)
              THEN Writeln (Pl, Pl_CharDir, 1, Sep, 0)
              ELSE Writeln (Pl, Pl_CharDir, 0, Sep, 1);
{           FontDiv   := 4 * (2 * SheetSizes[1] DIV 100);
           FontMult  := TRUNC (FontMagn * SheetSizes[1]/100 * 1.2);
           RelSize   := FontMult / FontDiv;}
           RelSize   := {0.8 *} 1.0 * FontMagn * 0.25;
{           Writeln (Pl, Pl_CharSize, RelSize * 0.19 :5:2, Sep, RelSize * 0.27 :5:2);}
           Writeln (Pl, Pl_CharRsize, RelSize * 0.75 :5:2, Sep, RelSize * 1.5 :5:2);
         END; {Pl_SetTextStyle}


     {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *}

     {Plotting - File Operators}


       PROCEDURE Pl_InitPlot;
         BEGIN
           ASSIGN (Pl, 'COM.txt');
           REWRITE (Pl);
           Pl_ResetViewPort;
           Writeln (Pl, Pl_PenThick, '0.5');
           Writeln (Pl, Pl_LineType, ';');
         END; {Pl_InitPlot}

       PROCEDURE Pl_EndPlot (Qty : Word);
         VAR
           i : Integer;
         BEGIN
           Writeln (Pl, Pl_Page);
           Writeln (Pl, Pl_Home);
           CLOSE (Pl);
           IF (Qty > 0)
              THEN FOR i := 1 TO Qty DO
                     BEGIN
                       SWAPVECTORS;
                       EXEC ('C:\COMMAND.COM', '/C MODE COM1: 9600,N,8,1,P');
                       EXEC ('C:\COMMAND.COM', '/C COPY COM.TXT COM1:');
                       SWAPVECTORS;
                     END; {Plot Qty of times - quality?}
         END; {Pl_EndPlot}

       PROCEDURE Pl_SavePlot (PathName : String);
         BEGIN
           SWAPVECTORS;
           EXEC ('C:\COMMAND.COM', '/C COPY COM.TXT ' + PathName);
           SWAPVECTORS;
         END; {Pl_SavePlot}


   END. {Pl_BGI}
