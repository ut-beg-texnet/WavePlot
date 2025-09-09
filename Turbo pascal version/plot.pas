{RESOURCE: Plotter Resource}

  UNIT Plot;

   {Contains data structures & operators, needed to plot the different graph types offered}

   {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }


   INTERFACE

     PROCEDURE Pl_PlotSeism;
     PROCEDURE PlotDump (Dmp : Word);
     PROCEDURE PlotSnap (Snp : Word);
     PROCEDURE Pl_WriteText;


    {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *}


   IMPLEMENTATION

     USES
       Crt, Graph, Funct, DspForm, Setts, Data_Rsc, Pl_BGI, Dsprim, Plprim, Dspl, Dos;


     {Redefine set_color command, to test for single-colour setting}
       PROCEDURE Pl_SelectColour (Pen : Integer);
         VAR
           P : Integer;
         BEGIN
           P := ((Pen-1) MOD 8) + 1;
           {Map becase of stupid colour system in conversion programs!!}
           CASE P OF
             1 : Pen := 1;  {Black}
             2 : Pen := 5;  {Blue}   {make 4 for Photoshop}
             3 : Pen := 2;  {Red}
             4 : Pen := 3;  {Green}
             5 : Pen := 7;  {Cyan}
             6 : Pen := 6;  {Magenta}
             7 : Pen := 8;  {Brown}
             8 : Pen := 1;  {Black}
           END;
           IF (Sets.SameColour)
              THEN Writeln (Pl, Pl_PenChange, Sep, 1)
              ELSE Writeln (Pl, Pl_PenChange, Sep, Pen);
         END; {Pl_SelectColour}


       PROCEDURE Pl_GetBlockSz (Xqty, Yqty  : Integer;
                                VAR  Sz, St : Pl_Point);
         BEGIN
           IF (Xqty > Yqty)
              THEN BEGIN
                     Sz[2] := (Pl_Vscale+1) DIV Xqty;
                     Sz[1] := (Pl_Hscale+1) DIV Xqty;
                     St[2] := ( (Pl_Vscale+1) MOD Xqty ) DIV 2;
                     St[1] := ( (Pl_Hscale+1) MOD Xqty ) DIV 2;
                   END
              ELSE BEGIN
                     Sz[2] := (Pl_Vscale+1) DIV Yqty;
                     Sz[1] := (Pl_Hscale+1) DIV Yqty;
                     St[2] := ( (Pl_Vscale+1) MOD Yqty ) DIV 2;
                     St[1] := ( (Pl_Hscale+1) MOD Yqty ) DIV 2;
                   END;
         END; {Pl_GetBlockSz}


     PROCEDURE Pl_Set2Dscale;
       VAR
         Mfact : Single;
       BEGIN
         WITH Scale2D DO
           BEGIN
             Mfact := exp(Magn/2*ln(2));
{               Pl_Xpx := Pl_Hscale;
                Pl_Ypx := Pl_Vscale;}
             Pl_Xpx := TRUNC (Pl_SnapWind.Sz[1] * Mfact);
             Pl_Ypx := TRUNC (Pl_SnapWind.Sz[2] * Mfact);
             IF (TrueGscale)
                THEN BEGIN
                       IF (Xsz > Ysz)
                          THEN Pl_Ypx := TRUNC (Pl_Ypx * Ysz / Xsz);
                       IF (Ysz > Xsz)
                          THEN Pl_Xpx := TRUNC (Pl_Xpx * Xsz / Ysz);
                     END;
{               Pl_Xoff := (Pl_Hscale - Pl_Xpx) DIV 2;
                Pl_Yoff := (Pl_Vscale - Pl_Ypx) DIV 2;}
             Pl_Xoff := (Pl_SnapWind.Sz[1] - Pl_Xpx) DIV 2;
             Pl_Yoff := (Pl_SnapWind.Sz[1] - Pl_Ypx) DIV 2;
           END;
       END; {Set2Dscale}


 {     PROCEDURE Pl_CalcScale (Xqty, Yqty : Integer);
       BEGIN
         IF ( (Pl_Hscale/Xqty) > (Pl_Vscale/Yqty) )
            THEN BEGIN
                   Pl_Pscale[y] := (Pl_Vscale+1) / Yqty;
                   Pl_Pscale[x] := (Pl_Hscale+1) / Yqty;
                 END
            ELSE BEGIN
                   Pl_Pscale[x] := (Pl_Hscale+1) / Xqty;
                   Pl_Pscale[y] := (Pl_Hscale+1) / Xqty;
                 END;
       END; {Pl_CalcScale}


     PROCEDURE Pl_Titles;
       BEGIN
         Pl_SelectColour (PlPen_Titles);
         Pl_SetWindow (Pl_HdTwind);
         Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
         Pl_SetTextJustify (CenterText, CenterText);
         Pl_OutTextXY (Pl_HdTwind.Sz[1] DIV 2, Pl_HdTwind.Sz[2] DIV 2, Sets.HdTitle);
         Pl_SetWindow (Pl_HzTwind);
         Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
         Pl_SetTextJustify (CenterText, CenterText);
         Pl_OutTextXY (Pl_HzTwind.Sz[1] DIV 2, Pl_HzTwind.Sz[2] DIV 2, Sets.HzTitle);
         Pl_SetWindow (Pl_VtTwind);
         Pl_SetTextStyle (SmallFont, VertDir, FontSz);
         Pl_SetTextJustify (CenterText, CenterText);
         Pl_OutTextXY (Pl_VtTwind.Sz[1] DIV 2, Pl_VtTwind.Sz[2] DIV 2, Sets.VtTitle);
       END; {Pl_Titles}


     PROCEDURE Pl_NamePlot;
       VAR
         Confirm, Num1, Num2 : Char;
       BEGIN
         REPEAT
           Ds_AddMessage (1,7, 'To Save Plot, Enter two');
           Ds_AddMessage (2,7, '  digits (m,n). File will ');
           Ds_AddMessage (3,7, '  be given extension .Pmn.');
           Ds_AddMessage (4,7, '  (Non-digits are ignored)');
           Num1 := ReadKey;
           Num2 := Readkey;
           IF ( (Num1 IN ['0'..'9']) AND (Num2 IN ['0'..'9']) )
              THEN BEGIN
                     Ds_AddMessage (6,7, 'Save as fn.p'+Num1+Num2);
                     Ds_AddMessage (7,7, '   Confirm (Y/N)? ');
                     Ds_WaitResponse (Confirm);
                     IF (Confirm IN ['Y','y'])
                        THEN Pl_SavePlot (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.P'+Num1+Num2);
                   END
              ELSE BEGIN
                     Ds_AddMessage (6,7, 'Do not Save Plot File');
                     Ds_AddMessage (7,7, '   Confirm (Y/N)? ');
                     Ds_WaitResponse (Confirm);
                   END;
         UNTIL (Confirm IN ['Y', 'y']);
       END; {Pl_NamePlot}

     PROCEDURE Pl_Descript;
       CONST
         Hinset = 200;
       VAR
         Ch      : Char;
         i, Divs : Word;
         Vstp    : LongInt;
         y, m, d, dow : Word;
       BEGIN
         Pl_SetWindow (Pl_XtrWind);
         Pl_SelectColour (PlPen_List);
         Vstp := Pl_XtrWind.Sz[2] DIV 9;
         Divs := 6;
         WITH Sets DO
         WITH Pl_XtrWind DO
           BEGIN

             Pl_SetTextJustify (LeftText, BottomText);

             Pl_SetTextStyle (SmallFont, HorizDir, FontSz+3);
             Pl_MoveTo ( 0, TRUNC ((Divs-2.5)/Divs * Pl_XtrWind.Sz[2]) );
             Writeln (Pl, Pl_PenThick, '1.0');
             Pl_OutText ('       WVPLOT');

             Pl_ResetLineStyle;
             Pl_SetTextJustify (LeftText, BottomText);
             Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
             Pl_MoveTo ( 0, TRUNC ((Divs-4.0)/Divs * Sz[2]) );
{             Pl_SetTextJustify (CenterText, CenterText);}
{             Pl_MoveTo ( Sz[1] DIV 2, TRUNC ((Divs-3.5)/Divs * Sz[2]) );}
             GetDate(y,m,d,dow);
             Pl_OutText ('        (v'+ver+','+'   '+strng(m,0)+'/'+strng(d,0)+ '/'+strng(y,0)+')  ');

             Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
             Pl_SetTextJustify (CenterText, CenterText);
             Pl_MoveTo ( Sz[1] DIV 2, TRUNC ((Divs-5.0)/Divs * Sz[2]) );
             Pl_OutText (Copy (Title[1], 1, 20) );


(*             Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
             Pl_SetTextJustify (CenterText, TopText);
             Pl_MoveTo ( Pl_XtrWind.Sz[1] DIV 2, 8*Vstp);
             Pl_OutText ('  (File: ' + Drive + ':' + Dir + '\' + Fname + ')' );*)

{             Pl_SetTextStyle (SmallFont, HorizDir, FontSz+1);
             Pl_SetTextJustify (CenterText, CenterText);
             Pl_MoveTo ( Pl_XtrWind.Sz[1] DIV 2, 6 * Vstp);
             Pl_OutText (Copy (Title[1], 1, 20) );}

{             Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
             Pl_SetTextJustify (LeftText, CenterText);
             Pl_MoveTo ( Hinset, 4 * Vstp);
             Pl_OutText (Copy (Title[2], 1, 30) );
             Pl_MoveTo ( Hinset, 3 * Vstp);
             Pl_OutText (Copy (Title[3], 1, 30) );
             Pl_MoveTo ( Hinset, 2 * Vstp);
             Pl_OutText (Copy (Title[4], 1, 30) ); }
           END; {With}
       END; {Pl_Descript}

     {Seismogram Routines}
     {$I Plt\PlSeism}

     {Snap Plot Routines}
     {$I Plt\PlSnap}

     {Dump Plot Routines}
     {$I Plt\PlDump}

   END. {Plot_Rsc}

